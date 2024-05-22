import sys,os
import pathlib
file_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(file_path,'utils'))
sys.path.append(os.path.join(file_path,'realistic_fl_noise'))

from realistic_fl_dataloader import OL_Full_Noise,collate_fn_OL

def make_dataset(dataset_name,input_name,device,testing,config_path,T,output_shape,leadingT=0):
    
    if dataset_name =="OL24":
        data_set=OL_Full_Noise(input_name+".yaml",device,testing=testing,T=T,config_path=config_path,output_shape =output_shape,leadingT=leadingT)
        collate = collate_fn_OL
        
    return data_set, collate