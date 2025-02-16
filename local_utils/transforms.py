import numpy as np
import torch
import scipy
from scipy.signal import iirnotch
import os

class BipolarMontage():
    def __init__(self,storage_channels, bipolar_channels):
        self.bipolar_ids = np.array([[storage_channels.index(channel.split('-')[0]), storage_channels.index(channel.split('-')[1])] for channel in bipolar_channels])
    def __call__(self,signal):
        signal = signal[self.bipolar_ids[:,0]] - signal[self.bipolar_ids[:,1]]
        return signal

class AvgMontage():
    def __init__(self,storage_channels,avg_channels):
        avg_channels = [c.replace('-avg','') for c in avg_channels]
        self.avg_ids = np.array([storage_channels.index(channel) for channel in avg_channels])
    def __call__(self,signal):
        signal = signal[self.avg_ids]
        avg = signal.mean(axis=0).squeeze()
        signal = signal - avg
        return signal

class ReferentialMontage():
    def __init__(self,storage_channels,referential_channels):
        self.referential_ids = np.array([storage_channels.index(channel) for channel in referential_channels])
    def __call__(self,signal):
        signal = signal[self.referential_ids]
        return signal

class WindowCutter():
    def __init__(self,windowsize,jitter,Fq):
        self.windowsize = int(windowsize*Fq)
        self.jitter = int(jitter*Fq)

    def __call__(self,signal):
        # get index of center
        center = signal.shape[1]//2
        # get index of window start
        start = center - (self.windowsize)//2
        # shift by up -1 or 1 x offsetsize
        start = start + int(np.random.uniform(-1, 1)*self.jitter)
        return signal[:,start:start+self.windowsize]
        
def normalize(signal):
    signal = signal / (np.quantile(np.abs(signal), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
    signal = signal - np.mean(signal)
    
    return signal

class KeepRandomChannels():
    # init with list of signal montage channels, number of channels to be retained
    # use: input: signal
    # output: zero masked signal with *fixed* number of random channels retained
    def __init__(self,N_channels):
        self.N_channels = N_channels
    def __call__(self,signal):
        N_keeper = np.random.randint(1,self.N_channels)
        # choose n random channels
        keeper_indices = np.random.choice(self.N_channels, N_keeper, replace=False)
        output = np.zeros_like(signal)
        output[keeper_indices,:] = signal[keeper_indices,:]
        return output
    
class KeepNRandomChannels():
    # init with list of signal montage channels, number of channels to be retained
    # use: input: signal
    # output: zero masked signal with *fixed* number of random channels retained
    def __init__(self,N_channels,N_keeper):
        self.N_channels = N_channels
        self.N_keeper = N_keeper
    def __call__(self,signal):
        keeper_indices = np.random.choice(self.N_channels, self.N_keeper, replace=False)
        output = np.zeros_like(signal)
        output[keeper_indices,:] = signal[keeper_indices,:]
        return output
    
class KeepSelectedChannels():
    def __init__(self,montage_channels,keeper_channels):
        self.montage_channels = montage_channels
        self.keeper_channels = keeper_channels
    def __call__(self,signal):
        keeper_indices = np.array([self.montage_channels.index(channel) for channel in self.keeper_channels])        
        output = np.zeros_like(signal)
        output[keeper_indices,:] = signal[keeper_indices,:]
        return output

class SpikeDataset(torch.utils.data.Dataset):
    def __init__(self,df,path_folder, montage=None, windowcutter = None,transform=None,normalize=True,echo=True):
        # set lookup table
        self.df = df
        # set transform
        self.transform = transform
        # set windowcutter
        self.windowcutter = windowcutter
        # set montage
        self.montage = montage
        # set path to bucket
        self.path_folder = path_folder
        self.normalize = normalize
        if echo:
            if self.normalize: print('Dataloader normalizes signal!\n')
            else: print('Dataloader does not normalize singal!\n')

    def __len__(self):
        return len(self.df)

    def _preprocess(self,signal):
        # convert to desired montage
        signal = self.montage(signal)
        # cut window to desired shape
        signal = self.windowcutter(signal)
        # apply transformations
        if self.transform is not None:
            signal = self.transform(signal)                
        if self.normalize==True:
        # normalize signal
            signal = signal / (np.quantile(np.abs(signal), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
        # convert to torch tensor, the copy is due to torch bug
        signal = torch.FloatTensor(signal.copy())
        
        return signal

    def __getitem__(self, idx):
        # get name and label of the idx-th sample
        event_file = self.df.iloc[idx]['event_file']
        label = self.df.iloc[idx]['fraction_of_yes']
        # load signal of the idx-th sample
        path_signal = os.path.join(self.path_folder,event_file+'.npy')
        signal = np.load(path_signal)
        # preprocess signal
        signal = self._preprocess(signal)
        # return signal          
        return signal,label

class StandardFilter():
    """
    A preprocessing class designed for filtering and standardizing signals. 
    This class initializes with specific filtering parameters and applies 
    notch and bandpass filters to a given signal when called.

    Parameters:
        Fs (float): Target sampling frequency for the signal processing.
        Fs_orig (float): Original sampling frequency of the signal.
        notch_freq (float, optional): Central frequency to be notched out. Defaults to 60 Hz.
        band_low (float, optional): Lower cutoff frequency for the bandpass filter. Defaults to 0.5 Hz.
        band_high (float, optional): Upper cutoff frequency for the bandpass filter. Defaults to 65 Hz.

     Example:
        >>> preprocess = StandardPreprocess(Fs=128, Fs_orig=256, notch_freq=60, band_low=0.5, band_high=65)
        >>> filtered_signal = preprocess(raw_signal)
    """
    def __init__(self,Fs,notch_freq=60,band_low=0.5,band_high=60):
        self.Fs = Fs
        self.notch_b,self.notch_a = self.get_notch_param(Fs,notch_freq=notch_freq)
        self.band_b,self.band_a = self.get_band_param(Fs,band_low,band_high)


    def __call__(self,signal):
        signal = scipy.signal.lfilter(self.notch_b, self.notch_a, signal)
        signal = scipy.signal.filtfilt(self.band_b, self.band_a, signal) 
        signal[np.isnan(signal)] = 0
        return signal

    @staticmethod
    def get_notch_param(Fs,notch_freq,Q=30):
        # Design notch filter
        notch_freq = 60  # Frequency to remove
        Q = 30  # Quality factor
        b, a = iirnotch(notch_freq, Q, Fs)
        return b,a

    @staticmethod
    def get_band_param(Fs,lowcut,highcut,order=6):
        b, a = scipy.signal.butter(order, [lowcut, highcut], btype='bandpass', fs=Fs)
        return b,a 

