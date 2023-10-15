import random
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T
import torchvision.transforms.functional as TF

import clip

class CLIP:
    def __init__(self, device='cuda', name='ViT-B/16'):
        self.device = device
        self.name = name
        self.clip_model, self.transform_PIL = clip.load(self.name, device=self.device, jit=False)

        # disable training
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # image augmentation
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # placeholder
        self.text_zs = None
        self.image_zs = None
    
    def normalize(self, x):
        return x / x.norm(dim=-1, keepdim=True)
    
    def prepare_text(self, texts):
        # texts: list of strings.
        texts = clip.tokenize(texts).to(self.device)
        self.text_zs = self.normalize(self.clip_model.encode_text(texts))
        # print(f'[INFO] prepared CLIP text feature: {self.text_zs.shape}')
        return self.text_zs
    
    def prepare_image(self, images):
        images = self.transform(images)
        self.image_zs = self.normalize(self.clip_model.encode_image(images))
        return self.image_zs
    
    def __call__(self, input, mode='image') -> Any:
        if mode == 'image':
            return self.prepare_image(input)
        elif mode == 'text':
            return self.prepare_text(input)
        else:
            raise NotImplementedError

