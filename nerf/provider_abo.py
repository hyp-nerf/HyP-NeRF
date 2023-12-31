import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
import numpy as np
import random
import gzip
from scipy.spatial.transform import Slerp, Rotation
import os.path as osp

import trimesh

import torch
from torch.utils.data import DataLoader,Dataset

from .utils import get_rays


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose

class MetaNeRFDataset(Dataset):
    def __init__(self, opt, device, type='train', downscale=1, global_pose_index=0, n_test=10, class_choice = 'CHAIR') -> None:
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        if os.path.exists(os.path.join(opt.path, "data")):
            self.root_path = os.path.join(opt.path, "data")
        elif os.path.exists(os.path.join(opt.path, "ABO_rendered")):
            self.root_path = os.path.join(opt.path, "ABO_rendered")
        else:
            self.root_path = opt.path
        self.listing_path = os.path.join(opt.path, "ABO_listings/listings/metadata")
        self.n_test = n_test
        self.class_choice = class_choice
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.
        self.error_map = None
        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.global_pose_index = global_pose_index

        self.mode = 'abo'

        with gzip.open(os.path.join(self.listing_path,"listings_0.json.gz"), mode="r") as f:
            self.metadata = [json.loads(line) for line in f]
        for _listing in ['1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']:
            with gzip.open(os.path.join(self.listing_path,f"listings_{_listing}.json.gz"), mode="r") as f:
                for line in f:
                    json_dict= json.loads(line)
                    self.metadata.append(json_dict)
        
        types = []
        self.objects = os.listdir(self.root_path)
        self.objects.sort()
        for d in self.metadata:
            try:
                types.append(d['item_id'])
            except:
                print(d)

        print(f"Total objects in ABO Dataset: {len(types)}")
        filtered_objects = []
        print(f"Total rendered objects {len(self.objects)}")
        for val in self.objects:
            if val in types:
                if self.metadata[types.index(val)]['product_type'][0]['value'].lower() == self.class_choice.lower():
                    filtered_objects.append(val)

        self.objects = filtered_objects.copy()
        print(f"Total rendered {self.class_choice}s: {len(filtered_objects)}")
    
    def num_examples(self):
        return len(self.objects)
    
    def __len__(self):
        if self.type == 'test':
            return 200
        return len(self.objects)
    
    def get_obj_index(self, index):
        if self.type == 'test':
            self.index = torch.LongTensor([index])
            return index
        self.index = torch.randint(0,91,size=(1,))
        return index

    def get_random_rays(self,images,poses,intrinsics,H,W,index=None):
        if index is None:
            index = [0]
        B = len(index) # a list of length 1

        rays = get_rays(poses[index], intrinsics, H, W, self.num_rays, patch_size=self.opt.patch_size)
        results = {
            'H': H,
            'W': W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'patch_size': self.opt.patch_size,
        }

        if images is not None:
            images = images[index]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
            
            images_original = images.clone()
            image_clip = images_original[index].squeeze(0)
            image_clip = image_clip[..., :3] * image_clip[..., 3:]
            results['img_original'] = image_clip
               
        results['poses'] = poses
        results['intrinsics'] = intrinsics

        results['num_rays'] = self.num_rays

        return results
    
    def __getitem__(self, index, flag=None, view_pose_dir=None):
        index = self.get_obj_index(index)
        object_id = self.objects[index]
        
        data_path = os.path.join(self.root_path, object_id)
        if self.type == 'test':
            data_path = './'
        
        # load nerf-compatible format data.
        if self.mode == 'abo':
            with open(os.path.join(data_path, 'metadata.json'), 'r') as f:
                transform = json.load(f)
        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        H = W = 512
        
        # read images
        frames = transform["views"]

        poses = []
        images = None
        _iter = self.index.item()
        f = frames[_iter]
        if self.type != 'test':
            images = []
            f_path = os.path.join(data_path,'render','0',f'render_{_iter}.jpg')
            if not os.path.exists(f_path):
                f_path = os.path.join(data_path,'render','1',f'render_{_iter}.jpg')
            if not os.path.exists(f_path):
                f_path = os.path.join(data_path,'render','2',f'render_{_iter}.jpg')
            if not os.path.exists(f_path):
                print('path not found')
                exit()
            seg_path = os.path.join(data_path,'segmentation',f'segmentation_{_iter}.jpg')
            if self.mode == 'abo' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...
            
            image_without_mask = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
            mask = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            mask = np.expand_dims(mask, axis=-1)
            image = np.concatenate([image_without_mask.astype(np.float32),mask.astype(np.float32)],axis=-1)
            
            if H is None or W is None:
                H = image.shape[0] // self.downscale
                W = image.shape[1] // self.downscale

            # add support for the alpha channel as a mask.
            if image.shape[-1] == 3: 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

            if image.shape[0] != H or image.shape[1] != W:
                image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)

            # cv2.imwrite('dataset_test.png', image) # for debugging
            image = image.astype(np.float32) / 255.0 # [H, W, 3/4]
            images.append(image)
        
        pose = np.array(f['pose'], dtype=np.float32).reshape(4,4) # [4, 4]
        pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)
        poses.append(pose)
            
        poses = torch.from_numpy(np.stack(poses, axis=0)) # [N, 4, 4]

        if images is not None:
            images = torch.from_numpy(np.stack(images, axis=0)) # [N, H, W, C]
    
            if images.isnan().any() or images.isinf().any():
                print("Image has NaN or Inf")

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / self.downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / self.downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            fl_x = fl_y = 443.40496826171875 # focal length for ABO dataset

        cx = (transform['cx'] / self.downscale) if 'cx' in transform else (W / 2)
        cy = (transform['cy'] / self.downscale) if 'cy' in transform else (H / 2)
    
        intrinsics = np.array([fl_x, fl_y, cx, cy])
        results = self.get_random_rays(images, poses, intrinsics, H, W)

        results["filename"] = f"{object_id}_{self.index.item()}"
        return results, index
