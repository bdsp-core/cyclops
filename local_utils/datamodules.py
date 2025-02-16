import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
import os
import pandas as pd
import sys

# event dataset
class event_dataset():
    def __init__(self, path_files, Y, load_signal, signal_transforms=None):
        self.path_files = path_files
        self.event_files = [f.split('/')[-1] for f in self.path_files]
        self.Y = Y
        self.load_signal = load_signal
        self.signal_transforms = [] if signal_transforms == None else signal_transforms

    def __len__(self):
        return len(self.path_files)
    
    def __getitem__(self,idx):
        filepath = self.path_files[idx]
        y = self.Y[idx]
        x = self.load_signal(filepath)
        for transform in self.signal_transforms:
            x = transform(x)
        return x, y   

class DatamoduleBase(pl.LightningDataModule):
    def __init__(self,path_files,labels,modes,loader,batch_size=256,transforms=None,collate_fn=None):
        super().__init__()
        self.path_files = path_files
        self.event_files = [f.split('/')[-1].split('.')[0] for f in path_files]
        self.labels = labels
        self.loader = loader
        self.modes = modes
        self.transforms = [] if transforms ==None else transforms
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.prepare_data_per_node = True


    def build_dataloader(self,mode):
        path_files = self.get_path_files(mode)
        labels = self.get_labels(mode)
        shuffle = True if mode == 'Train' else False
        dataset = event_dataset(path_files,labels, self.loader, signal_transforms=self.transforms)
        dataloader = DataLoader(dataset,self.batch_size,shuffle=shuffle,num_workers=os.cpu_count(),persistent_workers=True, collate_fn=self.collate_fn)
        return dataloader

    def train_dataloader(self):        
        dataloader = self.build_dataloader('Train')
        return dataloader
    
    def val_dataloader(self):
        dataloader = self.build_dataloader('Val')
        return dataloader
    
    def test_dataloader(self):
        dataloader = self.build_dataloader('Test')
        return dataloader
    
    def get_labels(self,mode):
        return [label for label, m in zip(self.labels, self.modes) if m == mode]
    
    def get_path_files(self,mode):
        return [path_file for path_file, m in zip(self.path_files, self.modes) if m == mode]

    def get_event_files(self,mode):
        msk = np.array(self.modes) == mode
        event_files = np.array(self.event_files)[msk]
        return event_files
    
class np_loader():
    def __init__(self):
        pass
    def __call__(self,path_signal):
        signal = np.load(path_signal)
        signal = np.nan_to_num(signal)
        return signal
    
class datamoduleRep(DatamoduleBase):
    def __init__(self,batch_size=256,transforms=None):
        df = pd.read_csv('/Tables/split_representative_12FEB24.csv')
        loader = np_loader()
        path_files, labels, modes = [],[],[]
        for mode in ['Train','Val','Test']:
            df_mode = self.get_split_df(df,mode)
            path_files+=df_mode.path.to_list()
            labels+=df_mode.fraction_of_yes.to_list()
            modes+=[mode]*len(df_mode)
        super().__init__(path_files,labels,modes,loader,batch_size,transforms,collate_fn=collate_fn)
    
    def get_split_df(self,df,Mode,echo=False):
        if Mode == 'Train':
            df = df[df.Mode==Mode]
            df = df[(df.total_votes_received>=3)|(df.fraction_of_yes==0)]
        if Mode == 'Val':
            df = df[df.Mode==Mode]
            df = df[(df.total_votes_received>=3)|(df.fraction_of_yes==0)]
        if (Mode == 'Test'):
            df = df[(df.Mode==Mode)&(df.dataset=='center')]
            pos = df[(df.total_votes_received>=8)&(df.fraction_of_yes>=7/8)]
            neg = df[df.fraction_of_yes==0]
            N = min([len(pos),len(neg)])
            df = pd.concat([pos[:N],neg[:N]])
        df.reset_index(inplace=True)
        if echo:
            ratio = len(df[df.fraction_of_yes>0.5])/len(df)
            print(f'\n{Mode} with {len(df)} samples and label ratio pos/all: {np.round(ratio,2)}')
        return df

class datamoduleLocal(DatamoduleBase):
    def __init__(self,batch_size,transforms):
        df = pd.read_csv('../Tables/split_local_13FEB24.csv')
        path_files, labels, modes = [],[],[]
        for mode in ['Train','Val','Test']:
            df_mode = self.get_split_df(df,mode)
            path_files+=df_mode.path.to_list()
            labels+=df_mode.fraction_of_yes.to_list()
            modes+=[mode]*len(df_mode)
        loader = np.load

        super().__init__(path_files,labels,modes,loader,batch_size,transforms,collate_fn)

    def get_split_df(df,Mode,echo=False):
        if Mode == 'Train':
            df = df[df.Mode==Mode]
            df = df[(df.total_votes_received>=3)|(df.fraction_of_yes==0)]
        if Mode == 'Val':
            df = df[df.Mode==Mode]
            df = df[(df.total_votes_received>=3)|(df.fraction_of_yes==0)]
        if (Mode == 'Test'):
            df = df[(df.Mode==Mode)&(df.dataset=='center')]
            pos = df[~df.location.isna()]
            neg = df[df.fraction_of_yes==0]
            df = pd.concat([pos,neg])
        df.reset_index(inplace=True)
        if echo:
            ratio = len(df[df.fraction_of_yes>0.5])/len(df)
            print(f'\n{Mode} with {len(df)} samples and label ratio pos/all: {np.round(ratio,2)}')
        return df

class datamoduleClemson(DatamoduleBase):
    def __init__(self,batch_size,transforms=None):
        df = pd.read_excel('/media/moritz/Expansion/Data/Spikes_clemson_10s/tables/segments_labels_channels_montage.xlsx')
        path_folder = '/media/moritz/Expansion/Data/Spikes_clemson_10s/preprocessed_npy'
        path_files = [os.path.join(path_folder,event_file+'.npy') for event_file in df.event_file]
        labels = df.Spike.to_list()
        modes = ['Test']*len(path_files)
        loader = np.load
        super().__init__(path_files,labels,modes,loader,batch_size,transforms,collate_fn)
        
def collate_fn(batch):
    # process your batch
    X, y = zip(*batch)
    X = torch.tensor(np.array(X), dtype=torch.float32)  # Convert to float
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y