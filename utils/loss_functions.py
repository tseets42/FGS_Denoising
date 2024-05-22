import torch
from piqa import *
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchvision import models
from torch import nn
import torch.optim as optim
device = "cuda"

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.eps = epsilon
        self.MSE = nn.MSELoss(reduction="mean")
     
    def forward(self, img_recon,img_gt):
        return torch.sqrt(self.MSE(img_recon,img_gt) + self.eps**2)
    

class loss_func():
    def __init__(self,device,lams = [10.,30.], video_in=False,input_shape="t c h w",channels = 1): #gan_loss always weighted as 1.0?
        self.obj = ["min","max","max"]
        self.error_names = ["Charbonnier","PSNR","SSIM"]
        self.error_classes = [CharbonnierLoss(),PSNR(reduction="mean"),SSIM(reduction="mean",n_channels=channels)]
        self.errors = torch.zeros(len(self.error_classes),requires_grad=False)
        self.device = device
        self.N = 0
        self.input_shape = input_shape
        self.lams = lams
        self.loss_track = torch.tensor([[0,0,0]])
        self.video_in = video_in

        for i in range(len(self.error_classes)):
            self.error_classes[i] =self.error_classes[i].to(device)
        
            
    def get_batch_errors(self,img_recon,img_gt, img_wl,test_loss=False):
        img_recon=torch.nan_to_num(img_recon)
        
        weighted_loss = torch.tensor(0.,requires_grad=True).to(self.device)
        
        
        #reshape to make time dimension faster
        if self.video_in:
            
            img_recon = rearrange(img_recon, "b "+self.input_shape+" -> (b t) c h w")
            img_gt = rearrange(img_gt, "b "+self.input_shape+ "-> (b t) c h w")

        
       
        self.N += img_recon.shape[0]
        
        for i in range(len(self.error_classes)):
            loss_func = self.error_classes[i]
            if self.error_names[i]=="SSIM" or self.error_names[i]=="PSNR":
                loss = loss_func(torch.clip( img_recon,0,1),img_gt)
            else:
                loss = loss_func(img_recon,img_gt)
            
            sign = 1
            if self.obj[i] == "max":
                sign = -1
            weighted_loss += sign * loss*self.lams[i]
            
            self.errors[i] += loss.cpu().item()*img_recon.shape[0]
        return weighted_loss
            
    def get_mean_errors_and_reset(self):
        es = self.errors/(self.N+.00001)
        self.errors =torch.zeros(len(self.error_classes),requires_grad=False) 
        
        self.loss_track=torch.cat([self.loss_track,es.unsqueeze(dim=0)],dim=0)
        
        self.N=0
        return es
    
    def add_to_loss_tracking(self,losses):
        self.loss_track=torch.cat([self.loss_track,losses],dim=0)
    
    def get_loss_tracking(self):
        return self.loss_track[1:]
    
    def get_loss_names(self):
        return self.error_names
        
    def print_mean_errors(self):
        for i in range(len(self.error_classes)):
            print(self.error_names[i],": ",self.errors[i].item()/(self.N+.00001),", ",end="")
        print("",flush=True)
        
class loss_func2():
    def __init__(self,device,lams = [10.,30.],video_in=False,input_shape="t c h w",channels = 1): 
        
        self.obj = ["min","max","max","min","min","max","min","min","min","max","max"]
        self.error_names = ["Charbonnier","PSNR","SSIM","TV","LPIPS","MS_SSIM","GMSD","MS_GMSD","MDSI","HaarPSI","FSIM"]
        self.error_classes = [CharbonnierLoss(),PSNR(reduction="mean"),
                              SSIM(reduction="mean",n_channels=channels),
                              TV(reduction="mean"),
                              LPIPS(reduction="mean"),
                              MS_SSIM(reduction="mean",n_channels=channels),
                             GMSD(reduction="mean"),
                             MS_GMSD(reduction="mean"),
                             MDSI(reduction="mean"),
                             HaarPSI(reduction="mean"),
                             FSIM(reduction="mean")]
        self.errors =[]
        self.device = device
        self.N = 0
        self.input_shape = input_shape
        self.lams = lams
        self.loss_track = torch.tensor([[0 for i in range(len(self.error_classes))]])
        self.video_in = video_in

        for i in range(len(self.error_classes)):
            self.error_classes[i] =self.error_classes[i].to(device)

    def to_device(self,device):
        self.device = device
        for i in range(len(self.error_classes)):
            self.error_classes[i] =self.error_classes[i].to(device)
        
    def get_batch_errors(self,img_recon,img_gt, img_wl,test_loss=False):
        img_recon=torch.nan_to_num(img_recon)
        
        weighted_loss = torch.tensor(0.,requires_grad=True).to(self.device)
        
        #reshape to make time dimension faster
        if self.video_in:
            
            img_recon = rearrange(img_recon, "b "+self.input_shape+" -> (b t) c h w")
            img_gt = rearrange(img_gt, "b "+self.input_shape+ "-> (b t) c h w")

        
       
        self.N += img_recon.shape[0]
        error_list = []
        for i in range(len(self.error_classes)):
            loss_func = self.error_classes[i]
            #print(self.error_names[i])
            if self.error_names[i]=="TV":
                loss = loss_func(img_recon)
            elif self.error_names[i] in ["Charbonnier","PSNR","SSIM","MS_SSIM"]:
                loss = loss_func(torch.clip( img_recon,0,1),img_gt)
            else:
                img_recon_tmp = repeat(img_recon,"t c h w -> t (c r) h w",r =3)
                img_gt_tmp = repeat(img_gt,"t c h w -> t (c r) h w",r =3)
                loss = loss_func(torch.clip( img_recon_tmp,0,1),img_gt_tmp)
            
            sign = 1
            if self.obj[i] == "max":
                sign = -1
            weighted_loss += sign * loss*self.lams[i]
            error_list.append(loss.cpu().item()*img_recon.shape[0])
        if len(self.errors)==0:
            self.errors = torch.tensor([error_list])
        else:
            self.errors = torch.cat([self.errors,torch.tensor([error_list])],dim=0)
            #self.errors[i] += loss.cpu().item()*img_recon.shape[0]
        return weighted_loss
            
    def get_mean_errors_and_reset(self):
        es = self.errors.sum(dim=0)/(self.N+1e-6)
        self.errors =[]
        
        self.loss_track=torch.cat([self.loss_track,es.unsqueeze(dim=0)],dim=0)
        
        self.N=0
        return es
    
    def add_to_loss_tracking(self,losses):
        self.loss_track=torch.cat([self.loss_track,losses],dim=0)
    
    def get_loss_tracking(self):
        return self.loss_track[1:]
    
    def get_loss_names(self):
        return self.error_names
        
    def print_mean_errors(self):
        for i in range(len(self.error_classes)):
            print(self.error_names[i],": ",round(self.errors[:,i].sum().item()/(self.N+1e-6),4),", ",end="")
        print("",flush=True)      
