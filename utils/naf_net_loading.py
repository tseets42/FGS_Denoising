import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_naf_net_encoder(model,checkpoint):
    #load nafnet 32 
    with torch.no_grad():
        #intro embedding
        model.intro.weight.copy_(checkpoint['params']['intro.weight'])
        model.intro.bias.copy_(checkpoint['params']['intro.bias'])
        model.intro.requires_grad=False

        #down smaplers
        model.downs[0]=nn.Identity()
        for i in range(0,len(model.downs)-1):
            name = f'downs.{i}'
            model.downs[i+1].weight.copy_(checkpoint['params'][name+'.weight'])
            model.downs[i+1].bias.copy_(checkpoint['params'][name+'.bias'])
            model.downs[i+1].requires_grad=False

        #encoders
        for i in range(len(model.encoders)-1):
            name = f"encoders.{i}."
            temp_dict = {k[len(name):]: v for k, v in checkpoint['params'].items() if name in k}
            model.encoders[i].load_state_dict(temp_dict)
            model.encoders[i].requires_grad=False

        #middle block encoder
        name = f"middle_blks."
        temp_dict = {k[len(name):]: v for k, v in checkpoint['params'].items() if name in k}
        model.encoders[-1].load_state_dict(temp_dict)
        model.encoders[-1].requires_grad=False
        
        train_size=(1, 3, 256, 256)
        fast_imp=False
        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))
        
        with torch.no_grad():
            replace_layers(model, base_size=base_size, train_size=train_size, fast_imp=False,encoder=False)  
         
        model.no_grad_encoding = True
            
            
class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out

def replace_layers(model, base_size, train_size, fast_imp, encoder=False, **kwargs):
    encoder_start = encoder
    for n, m in model.named_children():
        if n=="encoders":
            encoder=True
        else:
            encoder=encoder_start
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, encoder, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d) and encoder:
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)