import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import OrderedDict
from torchmeta.modules import MetaSequential, MetaLinear

from metamodules import FCBlock, BatchLinear, HyperNetwork, get_subdict
from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer
from .clip_utils import CLIP



class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 bound=1,
                 log2_hashmap_size = 11,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(
            encoding, desired_resolution=2048 * bound,log2_hashmap_size=log2_hashmap_size)

        sigma_net = MetaSequential()
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.add_module(f"layer_{l}", BatchLinear(
                in_dim, out_dim, bias=False))
            if l != num_layers - 1:
                sigma_net.add_module(f"act_{l}", nn.ReLU(inplace=True))

        self.sigma_net = MetaSequential(sigma_net)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)

        color_net = MetaSequential()
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim_color

            color_net.add_module(f"layer_{l}", BatchLinear(
                in_dim, out_dim, bias=False))
            if l != num_layers_color - 1:
                color_net.add_module(f"act_{l}", nn.ReLU(inplace=True))

        self.color_net = MetaSequential(color_net)

        # background network
        if self.bg_radius > 0:
            raise NotImplementedError()
        else:
            self.bg_net = None

    def forward(self, x, d, mask=None, params=None):
        # sigma
        out = self.density(x,params)
        sigma,geo_feat = out['sigma'],out['geo_feat']

        # color
        color = self.color(x,d,mask,geo_feat,params)

        return sigma, color

    def density(self, x, params=None, idx=None):
        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound, idx=idx,params=get_subdict(params,'encoder.embeddings'))
        h = x
        
        h = self.sigma_net(h, params=get_subdict(params,'sigma_net'))
        
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def background(self, x, d):
        raise NotImplementedError()

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, params=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(
                mask.shape[0], 3, dtype=x.dtype, device=x.device)  # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h, params=get_subdict(params,'color_net'))

        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)  # fp16 --> fp32
        else:
            rgbs = h

        return rgbs


class HyPNeRF(nn.Module):
    '''
    Module class for HyP-NeRF model
    '''

    def __init__(self, opt, num_instances=1, mode='nerf', type='relu',
                 hn_hidden_features=512, hn_hidden_layers=1, hn_in=512, std=0.01, **kwargs):
        super().__init__()

        self.mode = mode
        self.num_instances = num_instances
        self.cuda_ray = opt.cuda_ray
        self.bg_radius = opt.bg_radius
        self.bound = opt.bound
        self.device = opt.device

        self.clip_mapping = opt.clip_mapping
        self.std = std

        self.shape_code = nn.Embedding(self.num_instances, hn_in)
        nn.init.normal_(self.shape_code.weight, mean=0, std=std)
        
        self.color_code = nn.Embedding(self.num_instances, hn_in)
        nn.init.normal_(self.color_code.weight, mean=0, std=std)

        if self.clip_mapping:
            self.clip_encoder = CLIP()
            self.clip_fc_shape = nn.Sequential(
                nn.Linear(hn_in , 512),
                nn.ReLU(True),
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.Linear(256, 256),
                nn.ReLU(True),
                nn.Linear(256, hn_in),
            )
            self.clip_fc_color = nn.Sequential(
                nn.Linear(hn_in , 512),
                nn.ReLU(True),
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.Linear(256, 256),
                nn.ReLU(True),
                nn.Linear(256, hn_in),
            )

        self.net = NeRFNetwork(
            encoding="hashgrid",
            bound=opt.bound,
            cuda_ray=opt.cuda_ray,
            density_scale=1,
            min_near=opt.min_near,
            density_thresh=opt.density_thresh,
            bg_radius=opt.bg_radius,
        )

        self.hyper_net = HyperNetwork(hyper_in_features=hn_in,
            hyper_hidden_layers=hn_hidden_layers,
            hyper_hidden_features=hn_hidden_features,
            hypo_module=self.net,activation=type)
        
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                print(name)


    def get_params(self, idx, input_dict):
        """
        """
        z_shape = self.shape_code(idx)
        z_color = self.color_code(idx)

        return self.hyper_net(z_shape, z_color)

    def run_clip_mapping(self, idx, input_dict):
        img_input = input_dict['img_original'].permute(0,-1,1,2)

        with torch.no_grad():
            clip_embedding = self.clip_encoder(img_input, mode="image").float()

        pred_shape = self.clip_fc_shape(clip_embedding)
        pred_color = self.clip_fc_color(clip_embedding)

        z_shape = self.shape_code(idx)
        z_color = self.color_code(idx)
        
        return {
            'pred_shape': pred_shape,
            'pred_color': pred_color,
            'shape_code': z_shape,
            'color_code': z_color
        }
    
    def forward(self, idx, input_dict, rays_o, rays_d, staged=False, bg_color=None, perturb=True, force_all_rays=False, test_finally=False, **kwargs):
        if self.clip_mapping:
            clip_output = self.run_clip_mapping(idx, input_dict)

            pred_shape, pred_color = clip_output['pred_shape'], clip_output['pred_color']
            gt_shape, gt_color = clip_output['shape_code'], clip_output['color_code']

            with torch.no_grad():
                # using the predicted mappings to generate the data
                pred_params = self.hyper_net(pred_shape, pred_color) 

                pred_rendered_output = self.net.render(rays_o, rays_d, staged=staged, bg_color=bg_color, perturb=perturb, force_all_rays=force_all_rays,params=pred_params,idx=idx, **kwargs)

                # using the hn mappings to generate the hn gt data
                gt_params = self.hyper_net(gt_shape, gt_color) 

                gt_rendered_output = self.net.render(rays_o, rays_d, staged=staged, bg_color=bg_color, perturb=perturb, force_all_rays=force_all_rays,params=gt_params,idx=idx, **kwargs)
            
            return [pred_rendered_output, gt_rendered_output, clip_output] 
        else:
            outputs = self.get_params(idx, input_dict)

            nerf = self.net.render(rays_o, rays_d, staged=staged, bg_color=bg_color, perturb=perturb, force_all_rays=force_all_rays, params=outputs, idx=idx, **kwargs)
            
            return nerf

