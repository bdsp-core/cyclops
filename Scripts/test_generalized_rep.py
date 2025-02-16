import json
import sys
sys.path.append('../')
from local_utils import ResNetInstance, TransformerInstance, SpikeNetInstance
from local_utils import WindowCutter, ReferentialMontage, KeepNRandomChannels
from local_utils import all_referential
import pytorch_lightning as pl
import numpy as np
import torch
import os 
import pandas as pd
from local_utils import datamoduleRep
from sklearn.metrics import roc_curve, auc

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

def init_transforms(montage_channels,storage_channels,windowsize,windowjitter,Fs,N_keeper):
    montage = ReferentialMontage(storage_channels,montage_channels)
    cutter = WindowCutter(windowsize,windowjitter,Fs)
    channel_remover = KeepNRandomChannels(N_channels=len(montage_channels),N_keeper=N_keeper)
    return [montage,cutter,channel_remover]

def generate_predictions(model,trainer,dataloader):
   preds = trainer.predict(model,dataloader)
   preds = np.concatenate(preds).squeeze()
   return preds

def load_params(path_model):
    with open(os.path.join(path_model,'params.json'), 'r') as fp:
        params = json.load(fp)
    return params

if __name__ == '__main__':
    path_model = '../Models/generalized_SpikeNet/'
    path_weights = os.path.join(path_model,'weights.ckpt')
    params = load_params(path_model)
    storage_channels = all_referential
    windowsize = 1
    batch_size = 128
    windowjitter = 0
    Fs = 128
    trainer = pl.Trainer(max_epochs=300, devices=1)

    model = load_model(params['architecture'],path_weights,params['n_channels'])
    n_runs = 5
    results = {'n_keeper':[],'run':[],'AUROC':[]}
    for N_keeper in range(params['n_channels']+1):
        print(f'>>> N keeper: {N_keeper}/{params["n_channels"]} <<<')
        for run in range(n_runs):
            transforms = init_transforms(params['montage_channels'],storage_channels,params['windowsize'],windowjitter=0,Fs=params['Fs'],N_keeper=N_keeper)
            module = datamoduleRep(batch_size,transforms)
            dataloader = module.test_dataloader()
            preds = generate_predictions(model,trainer,dataloader)
            
            labels = module.get_labels('Test')
            labels = np.array(labels).round(0).astype(int)
            fpr, tpr, thresholds = roc_curve(labels, preds)
            roc_auc = auc(fpr, tpr)

            results['n_keeper'].append(N_keeper)
            results['run'].append(run)
            results['AUROC'].append(roc_auc)
        
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(path_model,'results_deleted.csv'),index=False)

