# FGS Denoising
This repo is the code for "Video Denoising in Fluorescence Guided Surgery."

# Getting Started 
## Install
Install Requirements: This repo runs on a linux install with python=3.9.6, cuda=11.6.2
The linux enviorment can be installed with conda:
```
conda env create -f environment.yml
conda activate guided
```
## Training
This repo includes code for our 3 baseline models, a model can be trained by running:

```
python train_model.py -epochs 3000 -cpus 16 -batch 4 -model BL_RNN -input ol-2024-condor
```

The model configuration included for trianing are in model_configs/models


## Testing
Testing can be done using the following command:

```
python test_model.py -cpus 6 -batch 1 -input ol-2024-condor-no_crop -test test_models_5by6
```

To change the settings for testing a new file config file can be added to model_configs/testing

