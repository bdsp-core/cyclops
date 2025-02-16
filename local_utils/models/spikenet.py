import torch.nn as nn
import numpy as np
import torch

class FirstConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,dropout):
        # input shape [none, channels, ts, eeg_channels]
        # output shape 
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(8, 1), stride=(1, 1), padding=(4, 0), bias=False)  # Padding is manually set to emulate 'same'
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(8, 1), stride=(1, 1), padding=(4, 0), bias=False)  # Adjust padding
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.leaky_relu1 = nn.LeakyReLU(0.3)
        self.max_pooling1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu1(x)
        x = self.max_pooling1(x)
        x = self.dropout1(x)
            
        return x

class SecondConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,n_eeg_channels,dropout):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, n_eeg_channels), stride=(1, 1), padding=0, bias=False)  # 'valid' padding in PyTorch is equivalent to padding=0
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0), bias=False)  # Manual padding for 'same'
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.leaky_relu2 = nn.LeakyReLU(0.3)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x)
        return x
    
class OtherConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,dropout,downsample=False):
        super().__init__()
        self.downsample = downsample
        self.conv = nn.Conv2d(in_channels,out_channels, kernel_size=(9,1), stride=(1, 1), padding=(4,0), bias=False)  # 'valid' padding in PyTorch is equivalent to padding=0
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.3)
        if downsample:
            self.max_pooling = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        if self.downsample:
            x = self.max_pooling(x)
        x = self.dropout(x)
        return x

class DenseLayer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.dense = nn.Linear(in_channels,out_channels)

    def forward(self,x):
        x = self.dense(x)
        return x 
    
class SpikeNetHead(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.dense = DenseLayer(in_channels,out_channels)
        self.sigmoid = torch.sigmoid

    def forward(self,x):
        x = x.reshape(x.shape[0],-1)
        x = self.dense(x)
        x = torch.sigmoid(x)
        return x
    
class SpikeNet(nn.Module):
    def __init__(self,n_eeg_channels,dropout=0.2):
        super().__init__()
        self.block1 = FirstConvBlock(in_channels=1,out_channels=32,dropout=dropout)
        self.block2 = SecondConvBlock(in_channels=32,out_channels=64,n_eeg_channels=n_eeg_channels,dropout=dropout)
        self.block3 = OtherConvBlock(in_channels=64,out_channels=64,dropout=dropout,downsample=True)
        self.block4 = OtherConvBlock(in_channels=64,out_channels=94,dropout=dropout,downsample=False)
        self.block5 = OtherConvBlock(in_channels=94,out_channels=94,dropout=dropout,downsample=True)
        self.block6 = OtherConvBlock(in_channels=94,out_channels=128,dropout=dropout,downsample=False)
        

    def forward(self,x):
        # input shape [batch_size, eeg_channels, timesteps]
        # bring into [batch_size, channels,timesteps,eeg_channels]
        x = x.permute(0,2,1)
        x = x.unsqueeze(1)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x
