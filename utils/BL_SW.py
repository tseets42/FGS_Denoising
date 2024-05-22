import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchvision import transforms
from NAFNet_standard import *

class NAFNet_expanded(nn.Module):#nafnet with no residual connections and an expanded output feature space for fastnaf

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],out_channels=4, drop_out_rate = 0.0):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan,drop_out_rate=drop_out_rate) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan,drop_out_rate=drop_out_rate) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan,drop_out_rate=drop_out_rate) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

       
        
    def forward(self, inp):            
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)     
            
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        #_, _, h, w = x.size()
        h = x.size()[-2]
        w = x.size()[-1]
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class BL_SW(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],chan_to_denoise=[0,1,2], drop_out_rate = 0.0,num_frames_combined = 3,lookback_window=3, middle_expansion=2):
        super().__init__()
        out_channels=len(chan_to_denoise)
        self.chan_to_denoise = chan_to_denoise
        ref_channels = img_channel-out_channels
        self.ref_chans = [i for i in range(img_channel) if i not in self.chan_to_denoise]
        self.lookback_window = lookback_window
        self.num_frames_combined = num_frames_combined
        
        naf1_out_dim = out_channels*middle_expansion*num_frames_combined
        self.nafnet1 = NAFNet_expanded(img_channel*num_frames_combined, width, middle_blk_num, enc_blk_nums, dec_blk_nums,naf1_out_dim, drop_out_rate)
        
        self.nafnet2 = NAFNet_expanded( (naf1_out_dim+ref_channels*num_frames_combined)*lookback_window
, width, middle_blk_num, enc_blk_nums, dec_blk_nums,out_channels, drop_out_rate)
        
        self.combine_time = Rearrange("b t c h w -> b (t c) h w")
        self.t_step = 25
    def forward2(self,inp):
        B, T, C, H, W = inp.shape
        
        #final_resid=inp[:,-1:,self.chan_to_denoise]
        feats = []
        
        for i in range(self.lookback_window):
            x = inp[:,i:i+self.num_frames_combined]
            
            resid = self.combine_time(inp[:,i:i+self.num_frames_combined,self.chan_to_denoise])
            if len(self.ref_chans)>0:
                ref_resid = self.combine_time(inp[:,i:i+self.num_frames_combined,self.ref_chans])
            x = self.combine_time(x)#b (t c) h w
            x = self.nafnet1(x) # b out_chan h w, first u-net
            x[:,:resid.size(1)] = x[:,:resid.size(1)] + resid #residual connection on some of the channels
            if len(self.ref_chans)>0:
                x = torch.cat([x,ref_resid],dim=1)
                
            if i == self.lookback_window-1:
                final_resid= x[:,resid.size(1)-len(self.chan_to_denoise):resid.size(1)]
                
            feats.append(x)
        assert(i+self.num_frames_combined==T) #check if wrong sized input is passed
        feats = torch.cat(feats,dim=1) #combine intermediate features
        feats = self.nafnet2(feats)#second u-net, now only one denoised frame as output (b, #den, h w)
        feats = feats+final_resid
        return feats.unsqueeze(dim=1)

    def forward(self, frame_stack):
        B, T, C, H, W = frame_stack.shape
        if T == self.num_frames_combined-1+self.lookback_window:
            return self.forward2(frame_stack)
        
        
        t_step = self.t_step
        in_dev = frame_stack.device
        out = []
        lT = self.num_frames_combined-1+self.lookback_window-1
        for t in range(0,frame_stack.shape[1],self.t_step):
            inp = frame_stack[:,t:t+t_step]
            if t < lT:
                sh = inp.shape
                sh = list(sh)
                sh[1]=lT-t

                pad_zeros = torch.zeros(sh).to(in_dev)
                inp = torch.cat([pad_zeros,inp],dim=1)
            else:
                inp = frame_stack[:,t-lT:t+t_step]
            #print(inp.shape)
            inp = inp.unfold(1, lT+1, 1)
           # print(inp.shape)
            inp= rearrange(inp, "b t c h w l -> (b t) l c h w")

            o  = self.forward2(inp)

            o = rearrange(o,  "(b t) 1 c h w -> b t c h w",b =B)
            out.append(o)
        out = torch.cat(out,dim=1)
        return out

    
    