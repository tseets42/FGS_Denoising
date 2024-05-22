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
from itertools import product
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from ruamel.yaml import YAML
import pathlib
sys.path.append('utils')
from loss_functions import *
from naf_net_loading import replace_layers

import random

from make_dataset import make_dataset
from make_models import make_model

import argparse
from ruamel.yaml import YAML
import pathlib

import matplotlib.pyplot as plt

device = "cuda"
import pandas as pd

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


#usage
#python test_model.py -cpus 6 -batch 1 -input ol-2024-condor-no_crop -test test_models_5by6

T=100


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser("parser")
parser.add_argument('-cpus',  type=int, default=5)
parser.add_argument('-batch',  type=int, default=1)
parser.add_argument('-input',  type=str, default="ol-2024-condor-no_crop")
parser.add_argument('-path',type=str,default="")
parser.add_argument('-test',type=str,default="test_models_5by6")
parser.add_argument('-name',type=str,default="")



args = parser.parse_args()


num_cpus = args.cpus
batch = args.batch
input_name = args.input   
base_path = args.path
test_group = args.test
name = args.name

config_path="model_configs/"
yaml=YAML()  
path = pathlib.Path(config_path+"testing/"+test_group+".yaml")
test_yaml = yaml.load(path)

yaml=YAML()  
path = pathlib.Path("model_configs/inputs/"+input_name+".yaml")
input_yaml = yaml.load(path)




inp_shape = "t c h w"
test_set, collate_fn = make_dataset(input_yaml["ShortName"],input_name,device,testing=True,config_path=config_path,T=T,output_shape =inp_shape)
test_loader = DataLoader(test_set, batch_size=batch, shuffle=False,num_workers=num_cpus,pin_memory=True,collate_fn=collate_fn,worker_init_fn=seed_worker,generator=g)


pt_files = np.array(test_yaml["Models"])
t_steps=test_yaml["t_steps"]

models = []
shapes = []

for pt in pt_files:
    loadpath = base_path + pt
    
    gbt,sh = load_nafnet(loadpath)
    models.append(gbt.to('cpu'))
    shapes.append(sh)

print("Done loading models")



save_name = test_yaml["Name"] +"_"+name
print(save_name)
br= test_yaml["br"]
ss=test_yaml["ss"]
if test_yaml["cross_product"]:
    iterables = [pt_files,ss, br]
    index = pd.MultiIndex.from_product(iterables, names=["model", "S","B"])
else:
    iterables = [pt_files,list(zip(ss,br))]
    index=pd.MultiIndex.from_arrays(np.transpose([[p,sr[0],sr[1]] for p, sr in pd.MultiIndex.from_product(iterables, names=["model", "SB"])]),names=["model", "S","B"])
    
df = pd.DataFrame(columns=["losses"],index = index)

for i in range(len(df)):
    df.iloc[i]["losses"] = loss_func2(device,[1 for i in range(13)],video_in=True,input_shape=inp_shape)
    

    
my_file = Path(base_path+save_name+".pkl")
start_ind=0
if my_file.is_file():
    print("loading from move_save: ",base_path+save_name+".pkl")
    df = pd.read_pickle(base_path+save_name+".pkl")
    for i in range(len(df)):
        df.iloc[i]["losses"].to_device(device)
    start_ind=df.iloc[0]["losses"].errors.shape[0]
    

for frame_num in range(start_ind,len(test_set)):
    
    data = test_set[frame_num]
    data = [d.unsqueeze(dim=0) for d in data]  
    
    tt = time()
    print("Vid num: ",frame_num,flush=True)
    
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    gen_background = True
    for s,r in product(ss,br):

        test_set.s_test = s
        test_set.b_test = s*r


        with torch.no_grad():
            frame_stack,frame_gt,frames_wl = add_noise_to_data(data,test_set,'cpu',gen_background=gen_background)
            gen_background = False
            
        with torch.no_grad():
            for model,sh,model_names,t_step in zip(models,shapes,pt_files,t_steps):
                model = model.to(device)
                o2= []
                for t in range(0,frame_stack.shape[1],t_step):
                    input_temp = rearrange(frame_stack[:,t:t+t_step],"b t c h w -> b "+sh)
                    o=model(input_temp.to(device)).to(device)
                    o = rearrange(o,"b "+sh+" -> b t c h w")
                    o2.append(o)
                o =torch.cat(o2,dim=1)
                loss_criteria=df.loc[model_names,s,r]["losses"]
                if frame_num == start_ind:
                    print(model_names)
                    print(o.shape)
                    print(frame_gt.shape)
                l = loss_criteria.get_batch_errors(o,frame_gt.to(device),frames_wl)

                model = model.to('cpu')
    print("Time: ",time()-tt,"s",flush=True)
    
    for i in range(len(df)):
        df.iloc[i]["losses"].to_device('cpu')
    df.to_pickle(base_path+save_name+".pkl")
    for i in range(len(df)):
        df.iloc[i]["losses"].to_device(device)
    
for i in range(len(df)):
    df.iloc[i]["losses"].to_device('cpu')
df.to_pickle(base_path+save_name+"_final.pkl")
            
        
