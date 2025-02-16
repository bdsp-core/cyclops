import pyedflib 
from scipy.signal import resample_poly
import numpy as np


class EDFLoader():
    def __init__(self,storage_channels,Fs_up=None,transforms=None):
        self.transforms= _validate_property(transforms)
        self.storage_channels = storage_channels
        self.Fs_up = Fs_up
    def __call__(self,path_file):
            # apply transforms
        signal, Fs_orig = self._read_edf(path_file)
        for transform in self.transforms:
            signal = transform(signal)
        if self.Fs_up !=None:
            signal = resample_poly(signal,up=self.Fs_up,down=Fs_orig,axis=1)
        return signal 

    def _read_edf(self,path_file):
        with pyedflib.EdfReader(path_file) as f:
            signal_labels = f.getSignalLabels()
            self._check_labels(self.storage_channels,signal_labels)

            # init empty signal 
            signal = np.zeros((len(self.storage_channels), f.getNSamples()[0]))
            # iterate over storage channels and add them to the signal
            for i,channel in enumerate(self.storage_channels):
                signal_idx = signal_labels.index(channel)
                signal[i, :] = f.readSignal(signal_idx)
            Fs_orig = f.getSampleFrequency(signal_idx)
            Fs_orig = int(np.round(Fs_orig))
        return signal, Fs_orig

    def _check_labels(self,storage_channels,signal_labels):
        if not set(storage_channels).issubset(signal_labels):
            raise ValueError(f"not all storage channels are present in file! \n"
                             f"storage channels: {self.storage_channels} \n"
                             f"signal_labels: {signal_labels}")

def _validate_property(property):
    """Ensure signal_configs is a list."""
    if property == None:
        return []
    if not isinstance(property, list):
        return [property]
    return property