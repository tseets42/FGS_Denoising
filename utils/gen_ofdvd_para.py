
import numpy as np
import os
import glob
import os.path
import sys
from time import time

import torch
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchvision import transforms
import torchvision

import pathlib
import cv2
import multiprocessing
import OFDV

def sub_process_gen_ofdvd(frame_stack,frame_gt,ind,samples_per_vid,b,num_frames):
    gt_out = []
    of_out = []
    counts_out = []
    wl_out=[]
    
    
    if ind ==-1:
        indexes = np.random.choice(100-num_frames,samples_per_vid)
        max_index = max(indexes)
    else:
        starting_index = ind
        indexes = [ind]
        max_index = ind
    

    wl = np.array((255*frame_stack[b,:max_index+num_frames,:,:,:-1].cpu()).numpy(),dtype=np.uint8)
    fl = rearrange(frame_stack[b,:max_index+num_frames,:,:,-1:],"t h w c -> t c h w").cpu().numpy()

    tcv = OFDV.OFDV()

    #5.8s per loop for 100 frames

    tcv.process_video( wl,fl)
    ofdv_out = tcv.get_fw_estimate()

    for starting_index in indexes:
        of_out.append(torch.tensor(ofdv_out[starting_index:starting_index+num_frames]).unsqueeze(dim=0))
        counts_out.append(torch.tensor(tcv.forward_fl[starting_index:starting_index+num_frames,-1]).unsqueeze(dim=0))

        wl_out_temp = frame_stack[b:b+1,starting_index:starting_index+num_frames,:,:,:-1]
        wl_out_temp=rearrange(wl_out_temp,"b t h w c -> b t c h w")

        wl_out.append(wl_out_temp)
        
        gt_out.append(frame_gt[b:b+1,starting_index+num_frames-1:starting_index+num_frames,:,:,0])
    wl_out = torch.cat(wl_out,dim=0)
    of_out = torch.cat(of_out,dim=0)
    counts_out = torch.cat(counts_out,dim=0)
    gt_out = torch.cat(gt_out,dim=0)
    return of_out, wl_out, counts_out, gt_out

def generate_ofdvd_training_input(frames_in,frames_ground_truth,ind=-1,samples_per_vid=2,num_frames=5):
    device = frames_in.device
    frames_in = frames_in.cpu()
    frames_ground_truth = frames_ground_truth.cpu()
    
    batches = frames_in.shape[0]
    input_list = [[frames_in,frames_ground_truth,ind,samples_per_vid,i,num_frames] for i in range(batches)]

    pool = multiprocessing.Pool(batches)

    of_out, wl_out, counts_out, gt_out = zip(*pool.starmap(sub_process_gen_ofdvd,  input_list))
    
    wl_out = torch.cat(wl_out,dim=0).to(device)
    of_out = torch.cat(of_out,dim=0).to(device)
    counts_out = torch.cat(counts_out,dim=0).to(device)
    gt_out = torch.cat(gt_out,dim=0).to(device)
    return of_out, wl_out, counts_out, gt_out




def sub_process_gen_ofdvd_testing(frame_stack,frame_gt,b):
    gt_out = []
    of_out = []
    counts_out = []
    wl_out=[]
    
    
    num_frames = 5

    wl = np.array((255*frame_stack[b,:,:,:,:-1].cpu()).numpy(),dtype=np.uint8)
    fl = rearrange(frame_stack[b,:,:,:,-1:],"t h w c -> t c h w").cpu().numpy()

    tcv = OFDV.OFDV()

    #5.8s per loop for 100 frames

    tcv.process_video( wl,fl)
    ofdv_out = tcv.get_fw_estimate()

    of_out.append(torch.tensor(ofdv_out).unsqueeze(dim=0))
    counts_out.append(torch.tensor(tcv.forward_fl[:,-1]).unsqueeze(dim=0))

    wl_out_temp = frame_stack[b:b+1,:,:,:,:-1]
    wl_out_temp=rearrange(wl_out_temp,"b t h w c -> b t c h w")

    wl_out.append(wl_out_temp)
    gt_out.append(frame_gt[b:b+1,:,:,:,0])
        
    wl_out = torch.cat(wl_out,dim=0)
    of_out = torch.cat(of_out,dim=0)
    counts_out = torch.cat(counts_out,dim=0)
    gt_out = torch.cat(gt_out,dim=0)
    return of_out, wl_out, counts_out, gt_out

def generate_ofdvd_testing_input(frames_in,frames_ground_truth):
    device = frames_in.device
    frames_in = frames_in.cpu()
    frames_ground_truth = frames_ground_truth.cpu()
    
    batches = frames_in.shape[0]
    input_list = [[frames_in,frames_ground_truth,i] for i in range(batches)]

    pool = multiprocessing.Pool(batches)

    of_out, wl_out, counts_out, gt_out = zip(*pool.starmap(sub_process_gen_ofdvd_testing,  input_list))
    
    wl_out = torch.cat(wl_out,dim=0).to(device)
    of_out = torch.cat(of_out,dim=0).to(device)
    counts_out = torch.cat(counts_out,dim=0).to(device)
    gt_out = torch.cat(gt_out,dim=0).to(device)
    return of_out, wl_out, counts_out, gt_out

