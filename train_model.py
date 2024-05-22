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

from pathlib import Path
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
import torchvision
sys.path.append('utils')
from loss_functions import *

import shutil
from make_dataset import make_dataset
from make_models import make_model, make_optimizer

def add_noise_to_data(data,data_set,device):
    frame_gt,frames_wl,read_noise = data #need to load then add noise because noise adding step uses a CNN 
    frames_wl = frames_wl.to(device)
    frame_gt = frame_gt.to(device)
    read_noise=read_noise.to(device)
    return data_set.add_random_noise(frame_gt,frames_wl,read_noise,gen_background = True)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


#usage:
# python train_model.py -epochs 3000 -multiGPU 0 -cpus 16 -batch 4 -model NAF_32_Standard -input ol-2024-condor -move /staging/seets/

parser = argparse.ArgumentParser("parser")
parser.add_argument('-epochs',  type=int, default=100)
parser.add_argument('-multiGPU',  type=int, default=0) #number of gpus
parser.add_argument('-cpus',  type=int, default=1)
parser.add_argument('-batch',  type=int, default=1)
parser.add_argument('-grad_checkpoint',  type=int, default=0)
parser.add_argument('-model',  type=str, default="")
parser.add_argument('-input',  type=str, default="ol-2024-condor")
parser.add_argument('-load',type=str,default="NONE")
parser.add_argument('-preload',type=str,default="NONE")
parser.add_argument('-move',type=str,default="models/")
parser.add_argument('-lr',type=float,default="0")
parser.add_argument('-name',type=str,default="")


args = parser.parse_args()
use_data_parallel = args.multiGPU > 0
args.world_size = args.multiGPU
num_epochs = args.epochs
use_data_parallel = args.multiGPU > 0
loadpath = args.load
load_model = (not (args.load=="NONE"))

loadpath = args.preload
preload_model = (not (args.preload=="NONE"))

move_save_name = args.move
move_save = (not (move_save_name=="NONE"))
learning_rate = args.lr
name_append = args.name

num_cpus = args.cpus
batch = args.batch
use_grad_checkpointing = args.grad_checkpoint > 0
model_name = args.model
input_name = args.input   
device = "cuda"


if load_model:
    print("Loading model from: ", loadpath)
    checkpoint = torch.load(loadpath,map_location='cpu')
    input_name=checkpoint['input_yaml']
    model_name=checkpoint['model_yaml']
    

yaml=YAML()   
path = pathlib.Path("model_configs/models/"+model_name+".yaml")
model_yaml = yaml.load(path)

yaml=YAML()  
path = pathlib.Path("model_configs/inputs/"+input_name+".yaml")
input_yaml = yaml.load(path)



#optimizer info
if learning_rate==0:
    learning_rate  = model_yaml["LearningRate"]


train_set, collate_fn = make_dataset(input_yaml["ShortName"],input_name,device,testing=False,config_path="model_configs/",T=model_yaml["max_T_train"],output_shape =model_yaml["input_shape"],leadingT=model_yaml["leadingT"])
train_loader = DataLoader(train_set, batch_size=batch, shuffle=True,num_workers=num_cpus,pin_memory=True,collate_fn=collate_fn)

test_set, collate_fn =make_dataset(input_yaml["ShortName"],input_name,device,testing=True,config_path="model_configs/",T=model_yaml["max_T_train"],output_shape =model_yaml["input_shape"],leadingT=model_yaml["leadingT"])
test_loader = DataLoader(test_set, batch_size=batch, shuffle=False,num_workers=num_cpus,pin_memory=True,collate_fn=collate_fn)

print("train length: ",len(train_loader))
print("test length: ",len(test_loader))


model_save_path = ""
Path(model_save_path).mkdir(parents=True, exist_ok=True)

model_type = model_yaml["Model"]

temp_h = input_yaml["h"]
temp_w = input_yaml["w"] 
save_name = model_name+"_"+f"h{  temp_h}_w{temp_w }_"+input_yaml["ShortName"]+f"_ep{num_epochs}_"+name_append
print("Saving files with name: ",save_name)

if move_save and not load_model:
    my_file = Path(move_save_name+save_name+".pt")
    if my_file.is_file():
        print("loading from move_save: ",move_save_name+save_name+".pt")
        loadpath = move_save_name+save_name+".pt"
        checkpoint = torch.load(move_save_name+save_name+".pt",map_location='cpu')
        load_model=True

def train(device,args):  
    gbt = make_model(model_type,model_yaml,input_yaml).to(device)

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

    optimizer = make_optimizer(model_type,model_yaml,input_yaml, learning_rate,gbt)
    

    loss_criteria = loss_func(device,[1.,0.,0.],video_in=True,input_shape=model_yaml["input_shape"],channels=len(input_yaml["chan_to_denoise"]))
    test_loss_criteria = loss_func(device,[1.,0.,0.],video_in=True,input_shape=model_yaml["input_shape"],channels=len(input_yaml["chan_to_denoise"]))

    #time stats
    print_times = True
    loading_time = 0
    gpu_move_time = 0
    forward_time = 0
    backward_time = 0 

    start_epoch=0
    
    if load_model or preload_model:
        print("Loading model checkpoint: ", loadpath)
        checkpoint = torch.load(loadpath,map_location='cpu')
        gbt.load_state_dict(checkpoint['model_state_dict'])
        if not preload_model:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']   
            if start_epoch > num_epochs:
                print("restarting at epoch 0")
                start_epoch =0
        loss_criteria.add_to_loss_tracking(checkpoint['train_loss'])
        test_loss_criteria.add_to_loss_tracking(checkpoint['test_loss'])
        gbt.train()
        gbt=gbt.to(device)
        
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs,eta_min=1e-7,last_epoch=start_epoch-1)

    iters = len(train_loader)
    
    print("Starting",flush=True)
    for epoch in range(start_epoch,num_epochs):  
        gbt.train()
        t = time()
        epoch_time = time()
        for i,data in enumerate(train_loader):
            if print_times:
                loading_time+=time()-t
                t=time()

            frame_stack,frame_gt,frames_wl = add_noise_to_data(data,train_set,device)

            if print_times:
                gpu_move_time+=time()-t
                t=time()

            optimizer.zero_grad()

            o = gbt(frame_stack.clone())

            if print_times:
                forward_time +=time()-t
                t=time()

            l = loss_criteria.get_batch_errors(o,frame_gt,frames_wl)
            l.backward()
            optimizer.step()

            if print_times:
                backward_time+=time()-t
                t=time()

            del frame_stack,frame_gt,frames_wl

        print("Epoch: ",epoch,", Time: ",time()-epoch_time,"s",flush=True,end="")
        
        #summary statistics
        loss_criteria.print_mean_errors()
        loss_criteria.get_mean_errors_and_reset() #loss_criteria adds errors to internal tracking list
        lr_sched.step()

        if print_times:
            print("\t Load Data: ",round(loading_time,2),"s, GPU Move: ",round(gpu_move_time,2),"s, Forward: ",round(forward_time,2),"s, backward: ",round(backward_time,2),"s")
        
        if (epoch %10==0):
            print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    
        if (epoch %5==0 and epoch > 0) or epoch == num_epochs-1:
            epoch_time = time()
            gbt.eval()
            with torch.no_grad():
                for data in iter(test_loader):
               
                    frame_stack,frame_gt,frames_wl = add_noise_to_data(data,test_set,device)
                    o = gbt(frame_stack.clone())
                    l = test_loss_criteria.get_batch_errors(o,frame_gt,frames_wl) 
                        
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
            
            if move_save:
                shutil.copyfile(save_name+".pt", move_save_name+save_name+".pt")
    torch.save({
                'epoch': epoch,
                'model_state_dict': gbt.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss_criteria.get_loss_tracking(),
                'train_loss': loss_criteria.get_loss_tracking(),
                'loss_labels':loss_criteria.get_loss_names(),
                'input_yaml':input_name,
                'model_yaml' : model_name
            }, save_name+"_final.pt")
    if move_save:
        shutil.copyfile(save_name+"_final.pt", move_save_name+save_name+"_final.pt")
    
    if use_data_parallel:
        dist.destroy_process_group()
    
if __name__=="__main__":
    print("Version 3-MS: Multi scale representation")
    print("Input Yaml: ", input_yaml["Name"])
    print("Model Yaml: ", model_yaml["Name"])
    print("Epochs: ",num_epochs)
    print("Use Multiple GPUS: ",use_data_parallel)
    print("Number CPUs:", num_cpus)
    print("Batch Size: ", batch)
    print("Grad Checkpointing",use_grad_checkpointing)
    
    if use_data_parallel:
        assert(1==0)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        mp.spawn(train, nprocs=args.world_size, args=(args,),join=True)
    else:
        train(device,args)
