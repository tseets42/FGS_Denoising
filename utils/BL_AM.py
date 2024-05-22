import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from einops import rearrange, reduce, repeat
from NAFNet_standard import NAFNet
from gen_ofdvd_para import generate_ofdvd_testing_input


class BL_AM(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1,enc_blk_nums=[], dec_blk_nums=[], resid=False,chan_to_denoise=[0], drop_out_rate = 0.0):
        super().__init__()
        self.nafnet = NAFNet(img_channel=5,
                             width=width,
                             middle_blk_num=middle_blk_num,
                             enc_blk_nums=enc_blk_nums,
                             dec_blk_nums=dec_blk_nums,
                             resid=resid,
                             chan_to_denoise=chan_to_denoise,
                             drop_out_rate=drop_out_rate)
        self.t_step = 10
        
    def forward(self, frames_fl, frames_wl=None, frames_counts= None ):
        if frames_wl is None or frames_counts is None:
            return self.forward3(frames_fl)
        else:
             return self.forward2(frames_fl, frames_wl, frames_counts)
    def forward2(self, frames_fl, frames_wl, frames_counts):
        B,T,H,W = frames_fl.shape
        frames_fl =rearrange(frames_fl,  "b t h w ->(b t) 1 h w")
        frames_wl = rearrange(frames_wl,  "b t c h w ->(b t) c h w")
        frames_counts =rearrange(frames_counts,  "b t h w ->(b t) 1 h w")
        x = torch.cat([frames_fl,frames_wl,frames_counts],dim=1)
        x = self.nafnet(x)
        x=rearrange(x,"(b t) 1 h w -> b t h w",b=B)
        return x
    
    def forward3(self,frame_stack):
        t_step = self.t_step
        frames_fl, frames_wl, frames_counts, _ = generate_ofdvd_testing_input(frame_stack,frame_stack)
        out = []
        for t in range(0,frame_stack.shape[1],self.t_step):
            o  = self.forward(frames_fl[:,t:t+t_step], frames_wl[:,t:t+t_step], frames_counts[:,t:t+t_step])
            o = rearrange(o,  "b (t c) h w -> b t h w c",c=1)
            out.append(o)
        out = torch.cat(out,dim=1)
        return out