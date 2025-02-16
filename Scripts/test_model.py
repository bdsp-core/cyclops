import sys
sys.path.append('../')
from local_utils import ResNetInstance, TransformerInstance, SpikeNetInstance
from local_utils import WindowCutter, BipolarMontage
from local_utils import all_referential, all_bipolar, all_average
import pytorch_lightning as pl
import numpy as np
import torch
import os 
import pandas as pd
from local_utils import datamoduleRep, datamoduleClemson, datamoduleHash, datamoduleLocal

def load_model(architecture,path_weights,n_channels):
    if architecture =='ResNet':
        model = ResNetInstance.load_from_checkpoint(path_weights,n_channels=n_channels,map_location=torch.device('cuda'))
    elif architecture =='Transformer':
        model = TransformerInstance.load_from_checkpoint(path_weights,n_channels=n_channels,map_location=torch.device('cuda'))
    elif architecture =='SpikeNet':
        model = SpikeNetInstance.load_from_checkpoint(path_weights,n_channels=n_channels,map_location=torch.device('cuda'))
    return model

def init_trainer():
   trainer = pl.Trainer(default_root_dir='./logging', enable_progress_bar=False,devices=1)
   return trainer

def init_transforms(montage_channels,storage_channels,windowsize,windowjitter,Fs):
    montage = BipolarMontage(storage_channels,montage_channels)
    cutter = WindowCutter(windowsize,windowjitter,Fs)
    return [montage,cutter]

def generate_predictions(model,trainer,dataloader):
   preds = trainer.predict(model,dataloader)
   preds = np.concatenate(preds).squeeze()
   return preds

def get_datamodule(dataset,batch_size,transforms):
    if dataset =='Rep':
        module = datamoduleRep(batch_size,transforms)
    elif dataset == 'Loc':
        module = datamoduleLocal(batch_size,transforms)
    elif dataset == 'Clemson':
        module = datamoduleClemson(batch_size,transforms)
    elif dataset == 'Hash':
        module = datamoduleHash(batch_size,transforms)
    else: 
        raise ValueError('Please specify dataset correctly! Options are: Rep, Loc, Clemson, Hash')
    return module

if __name__ == '__main__':
    dataset = 'Clemson' # Rep, Loc, Hash, Clemson are the options
    path_model = '../Models/SpikeNet/'
    path_weights = os.path.join(path_model,'weights.ckpt')
    architecture = 'SpikeNet'
    storage_channels = all_referential
    montage_channels = all_bipolar
    windowsize = 1
    windowjitter = 0
    Fs = 128
    n_channels = len(montage_channels)
    print(f'>>>{path_weights}<<<')
    model = load_model(architecture,path_weights,n_channels)
    transforms = init_transforms(montage_channels,storage_channels,windowsize,windowjitter,Fs)
    
    print('there is an increase added to the trasforms!')
    module = get_datamodule(dataset,batch_size=128,transforms=transforms)
    dataloader = module.test_dataloader()
    trainer = pl.Trainer(max_epochs=300, devices=1)
    preds = generate_predictions(model,trainer,dataloader)
    
    event_files, labels = module.get_event_files('Test'), module.get_labels('Test')
    result = pd.DataFrame({'event_file':event_files,'pred':preds,'label':labels})
    result.to_csv(os.path.join(path_model,'pred_'+dataset+'.csv'),index=False)
