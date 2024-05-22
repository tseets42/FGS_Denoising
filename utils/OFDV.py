# -*- coding: utf-8 -*-
"""
Modified from 

@InProceedings{pmlr-v227-seets24a, title = {OFDVDnet: A Sensor Fusion Approach for Video Denoising in Fluorescence-Guided Surgery}, author = {Seets, Trevor and Lin, Wei and Lu, Yizhou and Lin, Christie and Uselmann, Adam and Velten, Andreas}, booktitle = {Medical Imaging with Deep Learning}, pages = {1564--1580}, year = {2024}, editor = {Oguz, Ipek and Noble, Jack and Li, Xiaoxiao and Styner, Martin and Baumgartner, Christian and Rusu, Mirabela and Heinmann, Tobias and Kontos, Despina and Landman, Bennett and Dawant, Benoit}, volume = {227}, series = {Proceedings of Machine Learning Research}, month = {10--12 Jul}, publisher = {PMLR}, pdf = {https://proceedings.mlr.press/v227/seets24a/seets24a.pdf}, url = {https://proceedings.mlr.press/v227/seets24a.html} }
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm




def round_img(image_input,dtype=np.uint8):
    ## clip pixel values and casts to type given by dtype
    info = np.iinfo(dtype)
    image = np.clip(image_input,info.min,info.max)
    image = image.astype(dtype)
    return image
  
    

def opt_flow_farneback(img1,img2):
    #Returns optical flow between img1 and img2
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(img1_gray, img2_gray, None, 0.5, 23, 23, 3, 5, 1.2, 0)
    return flow    


def warp_from_flow(flow, to_warp_arr,inter = cv2.INTER_NEAREST):
    #warps each image along axis=0 of to_warp_arr according to flow 
    if to_warp_arr.ndim ==2:
        to_warp_arr = np.array([to_warp_arr])
    height, width = flow.shape[:2]
    R2 = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))
    pixel_map = (R2 + flow).astype(np.float32)
        
    to_return = np.zeros_like(to_warp_arr)
    for i in range(len(to_warp_arr)):
        to_warp = to_warp_arr[i]
        to_return[i,:,:] = cv2.remap(to_warp, pixel_map[..., 0], pixel_map[..., 1],inter)
    return to_return
        
def apply_mask(mask, to_mask_replace_ones,to_mask_replace_zeros=0):
    if to_mask_replace_ones.ndim ==2:
        to_mask_replace_ones = np.array([to_mask_replace_ones])
        
    if len(to_mask_replace_zeros) == 0:
        to_mask_replace_zeros = np.zeros_like(to_mask_replace_ones)
            
    if to_mask_replace_ones.shape != to_mask_replace_zeros.shape:
        print("Error in apply_mask: to_mask_replace_ones and to_mask_replace_zeros need to be same shape")
        return
            
    mask=np.round(np.clip(mask,0,1))
    if mask.ndim ==2:
        mask=np.tile(mask,(len(to_mask_replace_ones),1,1))
    return mask*to_mask_replace_ones+(1-mask)*to_mask_replace_zeros




class TwoChannelVideo:
    
    def __init__(self, flow_func=opt_flow_farneback,flow_mask_threshold=.07):
        
        self.white_light = []
        self.fluorescent = [] #fluorescent channel(s) can track multiple channels\
        
        #self.flow_func(img1,img2) is a function that calculates flow from img2->img1
        self.flow_func = flow_func
        self.flow_mask_threshold = flow_mask_threshold
    
    def get_mask_forward(self, next_white_light,curr_white_light,flow):
        next_gray = cv2.cvtColor(next_white_light, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_white_light, cv2.COLOR_BGR2GRAY)
        warped_curr = warp_from_flow(flow,curr_gray,cv2.INTER_CUBIC)
        oldMask=np.zeros_like(warped_curr)
        oldMask[np.abs(1-warped_curr/ (next_gray+1e-9))<self.flow_mask_threshold]=1
        return oldMask
    
    def deartifact_warped(self,next_white_light,next_fluorescent_noise,flow):
        mask = self.get_mask_forward(next_white_light,self.white_light,flow)
        
        self.fluorescent = apply_mask(mask, self.fluorescent,next_fluorescent_noise)
    
    def process_next_frame(self, next_white_light,next_fluorescent_noise):
            
        if next_fluorescent_noise.ndim ==2:
            next_fluorescent_noise = np.array([next_fluorescent_noise])
        
        next_fluorescent_noise = np.append(next_fluorescent_noise, [np.ones_like(next_fluorescent_noise[0])],axis=0)

        if len(self.white_light) == 0:
            self.white_light = next_white_light
            self.fluorescent = next_fluorescent_noise
            return
        
        #apply flow
        flow = self.flow_func(next_white_light,self.white_light)
        self.fluorescent = warp_from_flow(flow,self.fluorescent)+next_fluorescent_noise
        
        #deartifact - default is masking based on flow errors
        self.deartifact_warped(next_white_light,next_fluorescent_noise,flow)
        
        #update current frame
        self.white_light = next_white_light

    def fluorescent_estimate(self):
        #return emperical average of current frame
        return self.fluorescent[0]/self.fluorescent[-1]
             
        
class OFDV(TwoChannelVideo):
    def __init__(self, flow_func=opt_flow_farneback,flow_mask_threshold=.07):
        TwoChannelVideo.__init__(self, flow_func,flow_mask_threshold)
            
        self.forward_fl = []
        self.backward_fl = []
        self.fluorescent_frames_noise = []
        
    def process_video(self, white_light_frames,fluorescent_noise):
        
        self.white_light = []
        self.fluorescent = []
        self.forward_fl = []
        for i in range(len(white_light_frames)):
            self.process_next_frame(white_light_frames[i],fluorescent_noise[i])

            if len(self.forward_fl) ==0:
                self.forward_fl=[self.fluorescent]
            else:
                self.forward_fl= np.append(self.forward_fl,[self.fluorescent],axis=0)
                    
    def get_fw_estimate(self):
        return self.forward_fl[:,0]/self.forward_fl[:,-1]
    


