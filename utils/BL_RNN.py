import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

from NAFNet_standard import NAFNet


class BL_RNN(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],chan_to_denoise=[0,1,2], 
                 drop_out_rate = 0.0,resid=True,
                num_imgs = 2,
                use_layerNorm = True,
                use_channel_atn = True,
                use_simple_gate = True):
        super().__init__()
        self.chan_to_denoise =chan_to_denoise 
        self.num_imgs = num_imgs
        #refined image first as target to denoise
        self.nafnet = NAFNet(num_imgs*img_channel+num_imgs*len(chan_to_denoise), width, middle_blk_num, enc_blk_nums, dec_blk_nums,
                                      resid=resid,chan_to_denoise=[i for i in range(num_imgs*len(chan_to_denoise))], drop_out_rate =drop_out_rate,use_layerNorm = use_layerNorm, use_channel_atn = use_channel_atn, use_simple_gate = use_simple_gate)
        


    
    def nafnet_forward(self,x,z):
        #b t c h w
        z_out = self.nafnet(torch.cat([z,x],dim=2))
        return z_out
    
    
    def forward(self,x,z=None): #x,z
        #b t c h w
        B,T,C,H,W = x.shape
        if z is None:
            z = torch.zeros(x.shape).to(x)
            z = z[:,:,self.chan_to_denoise]
        #refined image first as target to denoise
        z_out = []
        
        start_index = self.num_imgs#different behavior for training and testing
        if not self.training:
            #print("test mode")
            start_index=1
        if start_index>=T+1:
            start_index=T
        for t in range(start_index,T+1):
            torch.cuda.empty_cache()
            z_in = z[:,t-1:t]
            if t == start_index:
                z_last = repeat(torch.zeros(z_in.shape).to(z_in),"b t c h w -> b (t r) c h w",r=self.num_imgs-1)
                
            if self.training or t>=self.num_imgs:  
                x_in = x[:,t-self.num_imgs:t]
            else:#causal at test time
                x_in = x[:,:t]
                x_in = torch.cat([repeat(x[:,:1],"b t c h w -> b (t r) c h w",r=self.num_imgs-t), x_in] ,dim=1)
                
            x_in = rearrange(x_in, "b t c h w -> b 1 (t c) h w")
            
            z_last = rearrange(z_last, "b t c h w -> b 1 (t c) h w")
            z_in = torch.cat([z_last,z_in],dim=2)
            
            z_pred = self.nafnet_forward(x_in, z_in)

            if t==self.num_imgs and self.training:
                z_out.append(rearrange(z_pred,"b 1 (t c) h w -> b t c h w",t=self.num_imgs))
            else:
                z_out.append(rearrange(z_pred,"b 1 (t c) h w -> b t c h w",t=self.num_imgs)[:,-1:])
            z_last = rearrange(z_pred,"b 1 (t c) h w -> b t c h w",t=self.num_imgs)[:,1:]

                
                
        return torch.cat(z_out,dim=1)
