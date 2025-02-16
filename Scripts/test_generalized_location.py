import json
import sys
sys.path.append('../')
from local_utils import SpikeNetInstance
from local_utils import WindowCutter, ReferentialMontage, KeepSelectedChannels
from local_utils import all_referential, all_bipolar, all_average
from local_utils import datamoduleHash, datamoduleClemson, datamoduleLocal
import pytorch_lightning as pl
import numpy as np
import torch
import os 
import pandas as pd
from local_utils import datamoduleRep
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm 

def load_model(path_weights,n_channels):
    model = SpikeNetInstance.load_from_checkpoint(path_weights,n_channels=n_channels,map_location=torch.device('cuda'))
    return model

def init_trainer():
   trainer = pl.Trainer(default_root_dir='./logging', enable_progress_bar=False,devices=1)
   return trainer

def init_transforms(montage_channels,storage_channels,windowsize,windowjitter,Fs,keeper_channels):
    montage = ReferentialMontage(storage_channels,montage_channels)
    cutter = WindowCutter(windowsize,windowjitter,Fs)
    channel_remover = KeepSelectedChannels(montage_channels,keeper_channels)
    return [montage,cutter,channel_remover]

def generate_predictions(model,trainer,dataloader):
   preds = trainer.predict(model,dataloader)
   preds = np.concatenate(preds).squeeze()
   return preds

def load_params(path_model):
    with open(os.path.join(path_model,'params.json'), 'r') as fp:
        params = json.load(fp)
    return params

def init_location_dict():
   # init channel and spike locations
   location_dict = {'frontal':['F3','F4'],
         'parietal':['P3','P4'],
         'occipital':['O1','O2'],
         'temporal':['T3','T4'],
         'central':['C3','C4'],
         'general':['Fp1','F3','C3','P3','F7','T3','T5','O1', 'Fz','Cz','Pz', 'Fp2','F4','C4','P4','F8','T4','T6','O2']}
   return location_dict

def get_datamodule(dataset,transforms):
    # get the right transforms and build dataset
    if dataset == 'Clemson':
        module = datamoduleClemson(transforms=transforms,batch_size=256)
    elif dataset == 'BonoboLocal':
        module = datamoduleLocal(transforms=transforms,batch_size=256)
    else:
        raise Exception('please specify dataset properly! Options: BonoboLocal, Clemson')
    return module

if __name__ == '__main__':
    dataset = 'Hash' # Rep, Loc, Hash, Clemson are the options
    path_model = '../Models/generalized_all_ref_loc/'
    path_weights = os.path.join(path_model,'weights.ckpt')
    params = load_params(path_model)
    storage_channels = all_referential
    windowsize = 1
    batch_size = 128
    windowjitter = 0
    Fs = 128
    trainer = pl.Trainer(max_epochs=300, devices=1)

    model = load_model(path_weights,params['n_channels'])

    location_dict = init_location_dict()
    results = {'event_file':[],'label':[],'pred':[],'ChannelLocation':[]}
    for ChannelLocation,keeper_channels in tqdm(location_dict.items()):
        transforms = init_transforms(params['montage_channels'],storage_channels,params['windowsize'],windowjitter=0,Fs=params['Fs'],keeper_channels=keeper_channels)
        
        module = get_datamodule(dataset, transforms)
        dataloader,labels,event_files = module.test_dataloader(), module.get_labels('Test'), module.get_event_files('Test')

        preds = generate_predictions(model,trainer,dataloader)

        results['event_file']+=list(event_files)
        results['label']+=list(labels)
        results['pred']+=list(preds)
        results['ChannelLocation']+=[ChannelLocation]*len(event_files)
    
results = pd.DataFrame(results)
results.to_csv(path_model+f'/results_localized_{dataset}.csv',index=False)