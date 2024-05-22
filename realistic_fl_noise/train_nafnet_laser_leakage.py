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
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.append('../utils')
from laser_leak_dataloader import *
from NAFNet_standard import NAFNet

sys.path.append('LLL_training')
from NAFNet_no_channel import NAFNet_no_chan


from naf_net_loading import replace_layers
from loss_functions import *
import argparse
from ruamel.yaml import YAML
import pathlib

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser("parser")
parser.add_argument('-epochs',  type=int, default=100)
parser.add_argument('-multiGPU',  type=int, default=0) #number of gpus
parser.add_argument('-cpus',  type=int, default=8)
parser.add_argument('-batch',  type=int, default=1)
parser.add_argument('-input',  type=str, default="laser_leak")
parser.add_argument('-model',  type=str, default="NAF_32_Standard_LLv2")
parser.add_argument('-fast', type=int, default=0)

parser.add_argument('-load',type=str,default="NONE")
parser.add_argument('-optim',type=str,default="ADAM")



args = parser.parse_args()
use_data_parallel = args.multiGPU > 0
args.world_size = args.multiGPU
num_epochs = args.epochs
use_data_parallel = args.multiGPU > 0
loadpath = args.load
load = (not (args.load=="NONE"))

num_cpus = args.cpus
batch = args.batch
model_name = args.model
input_name = args.input   
device = "cuda"
torch.cuda.device(device)

if load:
    print("Loading model from: ", loadpath)
    checkpoint = torch.load(loadpath,map_location='cpu')
    input_name=checkpoint['input_yaml']
    model_name=checkpoint['model_yaml']

yaml=YAML()   
path = pathlib.Path("../model_configs/models/"+model_name+".yaml")
model_yaml = yaml.load(path)

yaml=YAML()  
path = pathlib.Path("../model_configs/inputs/"+input_name+".yaml")
input_yaml = yaml.load(path)
#input info

ds=1
T=100
h=input_yaml["h"]//ds
w=input_yaml["w"]//ds
vid_size = [T,h,w]

#optimizer info
learning_rate  = model_yaml["LearningRate"]

#data loaders for train and test
data_path = input_yaml["InputPath"]
fast_train = (args.fast<=batch) and args.fast>0
print("Fast train: ",fast_train)

data_path = input_yaml["InputPath"]


noise_label = "laser_leak"
train_set = OL_Leakage(input_yaml["InputPath"],input_yaml["train_vids"],h=input_yaml["h"],w=input_yaml["w"],data_aug=input_yaml["Data_Aug"])
train_loader = DataLoader(train_set, batch_size=batch, shuffle=True,num_workers=num_cpus)

test_set = OL_Leakage(input_yaml["InputPath"],input_yaml["test_vids"],h=input_yaml["test_h"],w=input_yaml["test_w"])
test_loader = DataLoader(test_set, batch_size=batch, shuffle=False,num_workers=num_cpus)
    
model_save_path = ""
Path(model_save_path).mkdir(parents=True, exist_ok=True)

model_type = model_yaml["Model"]

def train(device,args):
    
    if model_type == "NAF_S":
        print("Using Standard NAFNet as base model")
        gbt = NAFNet(img_channel=model_yaml["Channels"], 
                       width=model_yaml["enc_dim"], 
                       middle_blk_num=model_yaml["middle_blk_num"], 
                       enc_blk_nums=model_yaml["enc_blocks"],
                       dec_blk_nums= model_yaml["dec_blocks"],
                       resid = model_yaml["resid"],
                       chan_to_denoise = model_yaml["chan_to_denoise"]).to(device)
        if model_yaml["pre-load32"]:
            load_path =model_yaml["load_path"]
            checkpoint = torch.load(load_path)
            gbt.load_state_dict(checkpoint['params'])
            train_size=(1, 3, 256, 256)
            fast_imp=False
            base_size = (int(h * 1.5), int(w * 1.5))
            replace_layers(gbt, base_size, train_size, fast_imp, encoder=True)
            gbt.train()    
    else:
        print("UNKNOWN MODEL NAME: ",model_type)
        return -1
        
        
    if use_data_parallel:
        dist.init_process_group(
        backend='nccl',
        init_method = 'env://',
        world_size = args.world_size,
        rank=device
    )

    gbt = gbt.to(device)

    if use_data_parallel:
        gbt = nn.parallel.DistributedDataParallel(gbt, device_ids=[device])
    
    if args.optim == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, gbt.parameters()), lr = learning_rate,momentum=0.9)
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, gbt.parameters()), lr = learning_rate)
    
    loss_criteria = loss_func(device,[1.,0.,0.],gan_loss=False)
    test_loss_criteria = loss_func(device,[1.,0.,0.],gan_loss=False)

    #time stats
    print_times = True
    loading_time = 0
    gpu_move_time = 0
    forward_time = 0
    backward_time = 0 

    start_epoch=0

    temp_h = input_yaml["h"]
    temp_w = input_yaml["w"] 
    path = ""
    save_name = path+model_name+"_"+f"_h{  temp_h}_w{temp_w }_"+noise_label+f"_ep{num_epochs}_L1_loss"
    print("Saving files with name: ",save_name)
    
    if load:
        gbt.load_state_dict(checkpoint['model_state_dict'])
        if args.optim == "ADAM":
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        gbt.train()
        gbt=gbt.to(device)
    
    print("Starting",flush=True)
    for epoch in range(start_epoch,num_epochs):
        t = time()
        epoch_time = time()
        for i,data in enumerate(train_loader):
            first_loop = True
            while fast_train or first_loop:
                first_loop = False
                if print_times:
                    loading_time+=time()-t
                    t=time()
                
                img_stack, img_gt = data
                img_stack = img_stack.to(device)
                img_gt = img_gt.to(device)

                if print_times:
                    gpu_move_time+=time()-t
                    t=time()
                
                optimizer.zero_grad()

                o = gbt(img_stack.clone())

                if print_times:
                    forward_time +=time()-t
                    t=time()
                    
                l = loss_criteria.get_batch_errors(o,img_gt,img_gt)
                
                l = (o-img_gt).abs().mean()
                
                l.backward()
                optimizer.step()



                if print_times:
                    backward_time+=time()-t
                    t=time()
                del img_stack,img_gt
                if fast_train:
                    loss_criteria.print_mean_errors()
                    loss_criteria.get_mean_errors_and_reset()

        print("Epoch: ",epoch,", Time: ",time()-epoch_time,"s",flush=True,end="")
        
        #summary statistics
        loss_criteria.print_mean_errors()
        loss_criteria.get_mean_errors_and_reset() #loss_criteria adds errors to internal tracking list
        

        if print_times:
            print("\t Load Data: ",round(loading_time,2),"s, GPU Move: ",round(gpu_move_time,2),"s, Forward: ",round(forward_time,2),"s, backward: ",round(backward_time,2),"s")
    
        if (epoch %5==0 and epoch > 0) or epoch == num_epochs-1:
            epoch_time = time()
            with torch.no_grad():
                for data in iter(test_loader):
                    img_stack, img_gt = data
                    img_stack=img_stack.to(device)
                    img_gt = img_gt.to(device)

                    o = gbt(img_stack.clone())
                    l = test_loss_criteria.get_batch_errors(o,img_gt,img_gt,test_loss=True)
                        
                        
            print("Test Loss, Time: ",time()-epoch_time,"s",flush=True,end="")
            test_loss_criteria.print_mean_errors()
            test_loss_criteria.get_mean_errors_and_reset()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': gbt.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss_criteria.get_loss_tracking(),
                'train_loss': loss_criteria.get_loss_tracking(),
                'loss_labels':loss_criteria.get_loss_names(),
                'input_yaml':input_name,
                'model_yaml' : model_name
                }, save_name+".pt")
            
            
    if use_data_parallel:
        dist.destroy_process_group()
    
if __name__=="__main__":
    print("Version 3-MS: Multi scale representation")
    print("Input Yaml: ", input_yaml["Name"])
    print("Model Yaml: ", model_yaml["Name"])
    print("Epochs: ",num_epochs)
    print("Number CPUs:", num_cpus)
    print("Batch Size: ", batch)
    
    if use_data_parallel:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        mp.spawn(train, nprocs=args.world_size, args=(args,),join=True)
    else:
        train(device,args)
