'''
Modified from 

Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchvision import transforms

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class OutputOne(nn.Module):
    def forward(self, x):
        return 1.0

    
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.,
                use_layerNorm = True,
                use_channel_atn = True,
                use_simple_gate = True):
        super().__init__()
        
        if use_simple_gate:
            dw_channel = c * DW_Expand
            ffn_channel = FFN_Expand * c
            ds_chan = 2
        else:
            dw_channel = c 
            ffn_channel =  c
            ds_chan=1
        
        
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // ds_chan, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // ds_chan, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        
        # Simplified Channel Attention
        if use_channel_atn:
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=dw_channel // ds_chan, out_channels=dw_channel // ds_chan, kernel_size=1, padding=0, stride=1,
                          groups=1, bias=True),
            )
        else:
            self.sca = OutputOne()

        # SimpleGate
        if use_simple_gate:
            self.sg = SimpleGate()
        else:
            self.sg = nn.ReLU()
            
        
        
        if use_layerNorm:
            self.norm1 = LayerNorm2d(c)
            self.norm2 = LayerNorm2d(c)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
        

class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], resid=False,chan_to_denoise=[0,1,2], drop_out_rate = 0.0,
                use_layerNorm = True,
                use_channel_atn = True,
                use_simple_gate = True
                ):
        super().__init__()

        out_channels=len(chan_to_denoise)
        self.chan_to_denoise = chan_to_denoise

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.resid = resid
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan,drop_out_rate=drop_out_rate,use_layerNorm = use_layerNorm, use_channel_atn = use_channel_atn, use_simple_gate = use_simple_gate) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan,drop_out_rate=drop_out_rate,use_layerNorm = use_layerNorm, use_channel_atn = use_channel_atn, use_simple_gate = use_simple_gate) for _ in range(middle_blk_num)]
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
                    *[NAFBlock(chan,drop_out_rate=drop_out_rate,use_layerNorm = use_layerNorm, use_channel_atn = use_channel_atn, use_simple_gate = use_simple_gate) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

        
    def get_encode(self, inp):
        T=0
        if len(inp.shape)==5:
            _, T, _, _, _ = inp.shape
            inp = rearrange(inp," b t c h w -> (b t) c h w")
            
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)
        
        if T>0:
            x = rearrange(x," (b t) c h w -> b t c h w",t=T)
            encs = [rearrange(e," (b t) c h w -> b t c h w",t=T) for e in encs]
            
        return x,encs
    
    
    def decode(self, inp, x, encs):
        T=0
        if len(inp.shape)==5:
            _, T, _, _, _ = inp.shape
            inp = rearrange(inp," b t c h w -> (b t) c h w")
            x = rearrange(x," b t c h w -> (b t) c h w")
            encs = [rearrange(e," b t c h w -> (b t) c h w") for e in encs]
        
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)


        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        
        if self.resid:
            x = x + inp[:,self.chan_to_denoise]
            
        if T>0:
            x = rearrange(x," (b t) c h w -> b t c h w",t=T)
            
            
        return x[:, :, :H, :W]
        
        
    def forward(self, inp):
        T=0
        if len(inp.shape)==5:
            _, T, _, _, _ = inp.shape
            inp = rearrange(inp," b t c h w -> (b t) c h w")
            
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
        
        if self.resid:
            x = x + inp[:,self.chan_to_denoise]
            
        if T>0:
            x = rearrange(x," (b t) c h w -> b t c h w",t=T)
            return x[:,:, :, :H, :W]
            
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        #_, _, h, w = x.size()
        h = x.size()[-2]
        w = x.size()[-1]
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    
    