import sys,os

import pathlib
file_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(file_path,'utils'))

from NAFNet_standard import NAFNet
from BL_SW import BL_SW
from BL_RNN import BL_RNN
from BL_AM import BL_AM

import torch.optim as optim

def load_ofdvd_nafnet(model,params):
    with torch.no_grad():

        #down smaplers
        for i in range(0,len(model.downs)):
            name = f'downs.{i}'
            model.downs[i].weight.copy_(params[name+'.weight'])
            model.downs[i].bias.copy_(params[name+'.bias'])

        #encoders
        for i in range(len(model.encoders)):
            name = f"encoders.{i}."
            temp_dict = {k[len(name):]: v for k, v in params.items() if name in k}
            model.encoders[i].load_state_dict(temp_dict)

        #middle block encoder
        name = f"middle_blks."
        temp_dict = {k[len(name):]: v for k, v in params.items() if name in k}
        model.middle_blks.load_state_dict(temp_dict)

        for i in range(len(model.decoders)):
            name = f"encoders.{i}."
            temp_dict = {k[len(name):]: v for k, v in params.items() if name in k}
            model.encoders[i].load_state_dict(temp_dict)
            
        for i in range(0,len(model.ups)):
            name = f'ups.{i}.0'
            temp_dict = {k[len(name):]: v for k, v in params.items() if name in k}
            model.ups[i][0].weight.copy_(params[name+'.weight'])
            #model.ups[i][0].bias.copy_(params[name+'.bias'])
        
            
        name = f"ending."
        temp_dict = {k[len(name):]: v for k, v in params.items() if name in k}
        model.ending.load_state_dict(temp_dict)

def make_model(model_type,model_yaml,input_yaml):
    if model_type =="BL_SW":
        print("Using BL_SW model")
        gbt  = BL_SW(img_channel=input_yaml["Channels"], 
                       width=model_yaml["enc_dim"], 
                       middle_blk_num=model_yaml["middle_blk_num"], 
                       enc_blk_nums=model_yaml["enc_blocks"],
                       dec_blk_nums= model_yaml["dec_blocks"],
                       chan_to_denoise = input_yaml["chan_to_denoise"],
                       drop_out_rate = model_yaml["drop_out_rate"],
                         num_frames_combined = model_yaml["num_frames_combined"],
                         lookback_window=model_yaml["lookback_window"], 
                         middle_expansion=model_yaml["middle_expansion"])
        
    elif model_type == "NAF_S":
        print("Using Standard NAFNet as base model")
        gbt = NAFNet(img_channel=input_yaml["Channels"], 
                       width=model_yaml["enc_dim"], 
                       middle_blk_num=model_yaml["middle_blk_num"], 
                       enc_blk_nums=model_yaml["enc_blocks"],
                       dec_blk_nums= model_yaml["dec_blocks"],
                       resid = model_yaml["resid"],
                       chan_to_denoise = input_yaml["chan_to_denoise"],
                       drop_out_rate = model_yaml["drop_out_rate"])
     
    elif model_type =="BL_AM":
        
        print("Using BL_AM as base model")
        gbt = BL_AM(img_channel=input_yaml["Channels"], 
                       width=model_yaml["enc_dim"], 
                       middle_blk_num=model_yaml["middle_blk_num"], 
                       enc_blk_nums=model_yaml["enc_blocks"],
                       dec_blk_nums= model_yaml["dec_blocks"],
                       resid = model_yaml["resid"],
                       chan_to_denoise = [0], #only takes input ofdv as first position....
                       drop_out_rate = model_yaml["drop_out_rate"])
        if model_yaml["pre_train"]:
            checkpoint = torch.load(model_yaml["nafnet_pretrained"],map_location='cpu')
            load_ofdvd_nafnet(gbt.nafnet,checkpoint['model_state_dict'])
        
    elif model_type =="BL_RNN":
        print("Using BL_RNN as base model")
        gbt = BL_RNN(img_channel=input_yaml["Channels"], 
                       width=model_yaml["enc_dim"], 
                       middle_blk_num=model_yaml["middle_blk_num"], 
                       enc_blk_nums=model_yaml["enc_blocks"],
                       dec_blk_nums= model_yaml["dec_blocks"],
                       resid = model_yaml["resid"],
                       chan_to_denoise = input_yaml["chan_to_denoise"],
                       drop_out_rate = model_yaml["drop_out_rate"],
                        num_imgs = model_yaml["num_imgs"],
                        use_layerNorm =  model_yaml["use_layerNorm"],
                        use_channel_atn = model_yaml["use_channel_atn"],
                        use_simple_gate = model_yaml["use_simple_gate"])
        
    else:
        print("UNKNOWN MODEL NAME: ",model_type)
        return -1
    return gbt




def make_optimizer(model_type,model_yaml,input_yaml, learning_rate,model):        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate)
        
    return optimizer