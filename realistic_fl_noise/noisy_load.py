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

import cv2


class OL_Noisy(Dataset):
    def __init__(self, parent_path,video_names,h,w,fps =15.0, output_shape = 't c h w',T=100,scale_fl=1.0):
        
        self.fps = fps
        self.h = h
        self.w = w

        self.video_file = []
        self.frameList = []
        for v in video_names:
            filename=parent_path + v
            vid = cv2.VideoCapture(filename)
            frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
            for i in range(0,int(frame_count),T):
                if i +T<frame_count:
                    self.video_file.append(filename)
                    self.frameList.append([i,i+T])
        self.scale_fl = scale_fl*1.0
        self.T = T
        self.resizing = transforms.Compose([Rearrange('t h w c -> '+ output_shape),
                                transforms.CenterCrop((self.h, self.w))])        
    def __len__(self):
        
        return len(self.video_file)
    
    def __getitem__(self, index):
        video_path = self.video_file[index]
        start_end = self.frameList[index].copy()            

        vid=torchvision.io.read_video(video_path,start_pts = start_end[0]/self.fps,end_pts=start_end[1]/self.fps ,pts_unit='sec')

        vid = vid[0][:start_end[1]-start_end[0]]
        
        assert(len(vid) == self.T)
        vid = vid.to(torch.float32)/255.0
        frames_wl = vid[:,768:,:1024]
        frame_fl = vid[:,:768,1024:,:1]*self.scale_fl

        frames_both = torch.cat((frames_wl,frame_fl),dim=3)
        
        frames_both = self.resizing(frames_both) 
        
        return frames_both
    
   