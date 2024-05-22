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


sys.path.append('../utils')
from NAFNet_standard import NAFNet
from naf_net_loading import replace_layers
from ruamel.yaml import YAML
import pathlib
import cv2
import matplotlib.pyplot as plt

def collate_fn_OL(batch):
    return (torch.stack([x[0] for x in batch]) ,torch.stack([x[1] for x in batch]),torch.stack([x[2] for x in batch]))

class OL_Full_Noise(Dataset):
    def __init__(self, input_yml_path,device,testing,config_path="model_configs/", output_shape = 't c h w',T=100,leadingT=0):
        
        yaml=YAML()   
        path = pathlib.Path(config_path+"inputs/"+input_yml_path)
        inp_yaml = yaml.load(path)
        
        self.parent_path = inp_yaml["InputPath"]
    
        if testing:
            set_dict = torch.load(self.parent_path+inp_yaml["train_test_split"])['test']
            self.R_path = self.parent_path + inp_yaml["R_path_test"]
            self.data_path = self.parent_path+"test"
            self.data_aug=False
        else:
            set_dict = torch.load(self.parent_path+inp_yaml["train_test_split"])['train']
            self.R_path = self.parent_path + inp_yaml["R_path_train"]
            self.data_path = self.parent_path+"train"
            self.data_aug = inp_yaml["data_aug"]
            
            
        self.frameList = set_dict["frames"]
        self.video = set_dict["videos"]
                
        self.fps = inp_yaml["fps"]
        
        self.B_min = inp_yaml["B_min"]
        B_network_path = self.parent_path+inp_yaml["B_network_path"]
        
        checkpoint = torch.load(B_network_path,map_location='cpu')
        model_name=checkpoint['model_yaml']

        yaml=YAML()   
        path = pathlib.Path(config_path+"models/"+model_name+".yaml")
        model_yaml = yaml.load(path)
        self.B_gen = NAFNet(img_channel=model_yaml["Channels"], 
                       width=model_yaml["enc_dim"], 
                       middle_blk_num=model_yaml["middle_blk_num"], 
                       enc_blk_nums=model_yaml["enc_blocks"],
                       dec_blk_nums= model_yaml["dec_blocks"],
                       resid = model_yaml["resid"],
                       chan_to_denoise = model_yaml["chan_to_denoise"]).to(device)

        checkpoint = torch.load(B_network_path,map_location='cpu')
        self.B_gen.load_state_dict(checkpoint['model_state_dict'])
        train_size=(1, 3, 256, 256)
        fast_imp=False
        base_size = (int(256 * 1.5), int(256 * 1.5))
        replace_layers(self.B_gen, base_size, train_size, fast_imp, encoder=True)
        self.B_gen = self.B_gen.eval()
        self.B_gen = self.B_gen.to(device)
        
        self.K_min = inp_yaml["K_inv_min"]
        self.K_max = inp_yaml["K_inv_max"]
        
        self.S_min = inp_yaml["S_min"]
        
        self.h = inp_yaml["h"]
        self.w = inp_yaml["w"]

        self.wl_only =inp_yaml["wl_only"]
        self.fl_only = inp_yaml["fl_only"]
        
        self.R_min = inp_yaml["R_min"]
        self.R_max = inp_yaml["R_max"]
        
            
        self.num_bits = inp_yaml["num_bits"]
    
        self.k_test = inp_yaml["k_inv_test"]
        self.r_test = inp_yaml["r_test"]
        self.s_test = inp_yaml["s_test"]
        self.b_test = inp_yaml["b_test"]
        self.device =device
        self.data_aug_v = inp_yaml["data_aug_v"]

        self.out_t = T
        self.T = T+leadingT
        
        
        self.output_reshape = Rearrange('b t c h w -> b '+ output_shape)
        self.standard_shape = Rearrange(' b '+ output_shape + ' -> b t c h w')
        
        self.t_step = inp_yaml["t_step"]
        self.background_images = None
        self.cap = cv2.VideoCapture(self.R_path)
        
        if self.data_aug:
            self.resizing = transforms.Compose([Rearrange('t h w c -> t c h w'),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomVerticalFlip(),
                                    transforms.RandomCrop((self.h, self.w))])
        else:
            self.resizing = transforms.Compose([Rearrange('t h w c -> t c h w'),
                                transforms.CenterCrop((self.h, self.w))])
    def __len__(self):
        
        return len(self.frameList)
    
    def generate_random_dark(self,num_frames):
        length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if length==0:
            print("dark frames video loading failed..")
            self.cap = cv2.VideoCapture(self.R_path)
            length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start = torch.randint(0, length-num_frames, (1,))[0].item()
        end = start+num_frames-1
        vid=torchvision.io.read_video(self.R_path,start_pts = start/self.fps,end_pts=end/self.fps ,pts_unit='sec')
        vid = vid[0][:num_frames]
        assert(len(vid) == num_frames)
        
        return vid/255.0
    
    def quantize(self,noise):
        scale = (2**self.num_bits-1)
        quantized = torch.round(noise*scale)/scale
        quantized = torch.clip(quantized,0,1)
        return quantized
    
    def reshape_output(self,frames):
        return  self.output_reshape(frames)
    
    def add_random_noise(self,frame_gt,frames_wl,read_noise,gen_background = True):
        #bached input!!!
        if self.data_aug:
            k_inv = torch.FloatTensor(1).uniform_(self.K_min, self.K_max)[0]
            r = torch.FloatTensor(1).uniform_(self.R_min, self.R_max)[0]
            
            if self.data_aug_v==0:
                s = torch.FloatTensor(1).uniform_(1, k_inv/2)[0]
                b = torch.FloatTensor(1).uniform_(0, s)[0]
            elif self.data_aug_v==3:
                s = torch.FloatTensor(1).uniform_(1, 100)[0]
                b = torch.FloatTensor(1).uniform_(25, 75)[0]
          
        else:
            k_inv=self.k_test
            r=self.r_test
            s=self.s_test
            b=self.b_test
        
        if self.wl_only:
            frame_stack = frames_wl
        else:
            frames_noise = self.add_noise(frame_gt,frames_wl, read_noise, k_inv, r, s, b,gen_background)

            if self.fl_only:
                frame_stack = frames_noise
            else:
                frame_stack = torch.cat([frames_wl,frames_noise],dim=2)
         
        #'b t c h w'        
        frame_gt = frame_gt[:,-self.out_t:]
        frames_wl = frames_wl[:,-self.out_t:]
        
        return self.reshape_output(frame_stack), self.reshape_output(frame_gt), self.reshape_output(frames_wl)
    
    def add_noise(self,frame_gt,ref,read_noise, k_inv, r, s, b,gen_background = True):
        #frames are batched!
        if not gen_background and self.background_images is not None: #skip for testing at different noise levels
            backgrounds = self.background_images
        else:
            with torch.no_grad():
                
                
                backgrounds=[]
                t_step =max(self.t_step//ref.shape[0],1)
                for t in range(0,ref.shape[1],t_step):
                    with torch.no_grad():
                        background = self.B_gen(ref[:,t:t+t_step].to(self.device))
                        backgrounds.append(background.to(frame_gt.device))
                backgrounds = torch.cat(backgrounds,dim=1)
                self.background_images = backgrounds 
        s_b = torch.clip(s*frame_gt+b*backgrounds,0)
        
        if torch.any(s_b<0):
            print("Failure to clip to 0 in add noise")
            s_b = torch.clip(s*frame_gt+b*backgrounds,1e-10)
            
        noise = torch.poisson( s_b ) /k_inv + read_noise[:,:,:1]/r
        noise = self.quantize(noise)*k_inv/s
        
        return noise
        
    def __getitem__(self, index):
        vid_id = int(self.video[index])
        start_end = self.frameList[index].copy()
        
        self.background_images = None
        
        if self.data_aug:
            if(start_end[1]-start_end[0]-self.T >0):
                start_end[0] = start_end[0]+torch.randint(0, start_end[1]-start_end[0]-self.T, (1,))[0].item()
                start_end[1] = start_end[0]+self.T           
        else:
            start_end[1] = min(start_end[1],start_end[0]+self.T)
            


        video_path = f"{self.data_path}/{vid_id}.mp4"
        
        vid=torchvision.io.read_video(video_path,start_pts = start_end[0]/self.fps,end_pts=start_end[1]/self.fps ,pts_unit='sec')

        vid = vid[0][:start_end[1]-start_end[0]]
        
        assert(len(vid) == self.T)
        vid = vid.to(torch.float32)/255.0
        frames_wl = vid[:,768:,:1024]
        frame_gt = vid[:,:768,1024:,:1]

        frames_both = torch.cat((frame_gt,frames_wl),dim=3) #channel dimension at end
        
        frames_both = self.resizing(frames_both) #data augmentation and resizing
        #(t c h w)
        frames_wl = frames_both[:,1:]
        frame_gt = frames_both[:,:1]

        read_noise = self.generate_random_dark(len(frame_gt))
        read_noise=self.resizing(read_noise)
        
        return frame_gt,frames_wl,read_noise
    
        