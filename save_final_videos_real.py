import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import os.path
import sys, getopt

from time import time
import torch
from torch import nn
import torch.nn.functional as F
import math

from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from piqa import *

import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from ruamel.yaml import YAML
import pathlib
sys.path.append('realistic_fl_noise')
sys.path.append('basic_vsr')
sys.path.append('basic_vsr/BasicVSR_PlusPlus')
sys.path.append('utils')
from loss_functions import *
from naf_net_loading import replace_layers

import random
#sys.path.append('realistic_fl_noise')
#from realistic_fl_dataloader import OL_Full_Noise,collate_fn

from make_dataset import make_dataset
from make_models import make_model

import argparse
from ruamel.yaml import YAML
import pathlib

import matplotlib.pyplot as plt
import torchvision

import cv2


def add_noise_to_data(data,data_set,device,gen_background=True):
    frame_gt,frames_wl,read_noise = data #need to load then add noise because noise adding step uses a CNN 
    frames_wl = frames_wl.to(device)
    frame_gt = frame_gt.to(device)
    read_noise=read_noise.to(device)
    return data_set.add_random_noise(frame_gt,frames_wl,read_noise,gen_background = gen_background)



#testing seed for reproducibility

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(11122345)

def load_nafnet(lpath,model_yaml=-1):
    
    checkpoint = torch.load(lpath,map_location='cpu')
    if model_yaml == -1:
        model_name=checkpoint['model_yaml']
    else:
        model_name = model_yaml
        
    if model_name == "BL_AM_preload":
        model_name = "BL_AM"
    
    yaml=YAML()   
    path = pathlib.Path(config_path+"models/"+model_name+".yaml")
    model_yaml = yaml.load(path)
    

    gbt = make_model(model_yaml["Model"],model_yaml,input_yaml)
    gbt.load_state_dict(checkpoint['model_state_dict'],strict=False)

    if "NAF" in model_yaml["Model"]:
        print("Updating Naf layers")
        train_size=(1, 3, 256, 256)
        fast_imp=False
        base_size = (int(256 * 1.5), int(256 * 1.5))
        replace_layers(gbt, base_size, train_size, fast_imp, encoder=True)
    gbt=gbt.to(device)
    gbt = gbt.eval()
    return gbt, model_yaml["input_shape"]


parser = argparse.ArgumentParser("parser")
parser.add_argument('-device',  type=int, default=1)

args = parser.parse_args()
device = f"cuda:{args.device}"
save_vid = "" #path to save videos

batch=1
num_cpus=10
T = 100
config_path="model_configs/"
input_name = "ol-2024-nova-no_crop"
yaml=YAML()  
path = pathlib.Path(config_path+"inputs/"+input_name+".yaml")
input_yaml = yaml.load(path)




from noisy_load import *

parent_path = "data/"
test_vids = ["real1.mp4","real2.mp4"]
test_set = OL_Noisy(parent_path,test_vids,h=768,w=1024,T=T,scale_fl=1.0)


path = "models/"

pt_files = np.array(["NAF_32_Standard_h256_w256_OL24_ep3000_lowlr_final.pt",
                    "BL_SW.pt",
                     "BL_AM.pt",
                     "BL_RNN.pt"
                    ])
names = ["Naf32","BL-SW","BL-AM","BL-RNN"]

models = []
shapes = []
for pt in pt_files:
    loadpath = path + pt
    
    gbt,sh = load_nafnet(loadpath)
    models.append(gbt.to('cpu'))
    shapes.append(sh)


models[1].t_step = 5 
models[2].t_step = 5 

t_steps=[5,100,100,100,100,100,100]
final_out_vid = []

for i in range(0,len(test_set)):
    vid_num = i
    print(f"vid_num_{vid_num}")
    with torch.no_grad():
        frame_stack = test_set[i].unsqueeze(dim=0)  

    outputs = []
    torch.cuda.empty_cache()
    with torch.no_grad():
        for model,sh,model_names,t_step in zip(models,shapes,pt_files,t_steps):
            torch.cuda.empty_cache()
            print(model_names,flush=True)
            model = model.to(device)
            model = model.eval()

            o2= []
            for t in range(0,frame_stack.shape[1],t_step):
                torch.cuda.empty_cache()
                input_temp = rearrange(frame_stack[:,t:t+t_step],"b t c h w -> b "+sh)
                o=model(input_temp.to(device)).to(device)
                o = rearrange(o,"b "+sh+" -> b t c h w")
                o2.append(o)
            print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024))
            o =torch.cat(o2,dim=1)
            outputs.append(o.to('cpu'))
            model = model.to('cpu')



    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 5
    fontColor              = (255,255,255)
    thickness              = 5
    lineType               = 2
    fls = frame_stack[0,:,-1:]
    img = fls/torch.max(fls)
    img = repeat(img, "t c h w -> t (c r) h w",r=3)

    wls = frame_stack[0,:,:-1]
    img = torch.cat([wls,img],dim=-1)
    img = rearrange(np.array(img.cpu().numpy()*255,dtype=np.uint8),"t c h w -> t h w c")
    text = np.zeros((img.shape[0],200,img.shape[2],3), np.uint8)
    img = np.append(text,img,axis=1)

    img=np.array([ cv2.putText(im,'Reference',  [150,150], font, fontScale,fontColor,thickness,lineType) for im in img ])
    img=np.array([ cv2.putText(im,'Noise',  [1300,150], font, fontScale,fontColor,thickness,lineType) for im in img ])

    all_outs = torch.clip(torch.cat(outputs,dim=-1)[0],0,1)

    all_outs = repeat(all_outs, "t c h w -> t (c r) h w",r=3)
    all_outs = rearrange(np.array(all_outs.cpu().numpy()*255,dtype=np.uint8),"t c h w -> t h w c")

    text = np.zeros((all_outs.shape[0],200,all_outs.shape[2],3), np.uint8)
    all_outs = np.append(all_outs,text,axis=1)

    loc = 100
    for pt in names:
        all_outs=np.array([ cv2.putText(im,pt[:10],  [loc,150+768], font, fontScale,fontColor,thickness,lineType) for im in all_outs ])
        loc = loc+1024
    pad = np.zeros(all_outs.shape,dtype=np.uint8)[:,:,img.shape[2]:,:]
    img = np.append(img,pad,axis=2)
    img=np.array([ cv2.putText(im,f'vid_num{vid_num}',  [3200,150], font, fontScale,fontColor,thickness,lineType) for im in img ])

    img = np.append(img,all_outs,axis=1)
    final_out_vid.append(img)

    if len(final_out_vid)>25:
        final_out_vid= np.array(final_out_vid)
        final_out_vid = rearrange(final_out_vid,"b t h w c -> (b t) h w c")
        torchvision.io.write_video(save_vid+f"part{vid_num}_real_data_scale2.avi", final_out_vid,15,video_codec ="h264",options={"qp":"0"})       
        final_out_vid = []


if len(final_out_vid)>0:
    final_out_vid= np.array(final_out_vid)
    final_out_vid = rearrange(final_out_vid,"b t h w c -> (b t) h w c")
    torchvision.io.write_video(save_vid+f"part{vid_num}_real_data_scale2.avi", final_out_vid,15,video_codec ="h264",options={"qp":"0"})
    final_out_vid = []
