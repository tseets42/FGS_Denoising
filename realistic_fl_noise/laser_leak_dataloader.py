import numpy as np
import os
import glob
import os.path
import sys
from time import time

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision import transforms
import torchvision

from torch.utils.data import Dataset, DataLoader

    

class OL_Leakage(Dataset):
    def __init__(self, parent_path,video_names,h,w,fps =15.0, data_aug = False):
        
        self.fps = fps
        self.h = h
        self.w = w
        self.data_aug = data_aug

        self.vids = torch.cat([torchvision.io.read_video(parent_path + v,pts_unit="sec")[0] for v in video_names],dim=0)
        
    def __len__(self):
        
        return len(self.vids)
    
    def __getitem__(self, index):
        
        
        if self.data_aug:
             resizing = transforms.Compose([Rearrange("b h w c -> b c h w"),transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.RandomRotation(35),
                                transforms.RandomResizedCrop((self.h, self.w), scale=(.1,1.0))])
        else:
            resizing = transforms.Compose([Rearrange("b h w c -> b c h w"),transforms.CenterCrop((self.h, self.w))])

        
        vid = self.vids[index]
        vid = vid.to(torch.float32)/255.0
        frame_wl = vid[768:,:1024]
        frame_fl = vid[:768,1024:]
        frame_stack = torch.cat([frame_wl.unsqueeze(dim=0),frame_fl.unsqueeze(dim=0)],dim=0)
        frame_stack=resizing(frame_stack)
        
        return frame_stack[0], frame_stack[1,:1,:,:] 
    
   