from pytorch_lightning import LightningModule
from torch.nn import BCELoss
import torch.nn as nn
import torch
from torch.nn import BCELoss
from .heads import RegressionHead
from .Spikenet import SpikeNet

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,target,pred):
        return torch.sqrt(self.mse(target,pred))

# create an instance base class
class InstanceBaseClass(LightningModule):
    def __init__(self,model,head,lr,weight_decay):
        super().__init__()
        self.lr = lr
        self.weight_decay=weight_decay
        # create the BIOT encoder
        self.model = model
        # create the regression head
        self.head = head
        self.RMSE = RMSELoss()
        self.loss = BCELoss()

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x

    def training_step(self,batch,batch_idx):
        x, target = batch
        # flatten label
        target = target.view(-1, 1).float()
        pred = self.forward(x)
        loss = self.loss(pred, target)
        self.log('train_loss', loss,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log('train_RMSE', self.RMSE(target=target,pred=pred),prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x, target = batch
        # flatten label
        target = target.view(-1, 1).float()
        pred = self.forward(x)
        loss = self.loss(pred, target)
        self.log('val_loss', loss,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log('val_RMSE', self.RMSE(target=target,pred=pred),prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        return loss
    
    def predict_step(self,batch,batch_idx,dataloader_idx=0):
        signals, labels = batch
        # flatten label
        labels = labels.view(-1, 1).float()
        # generate predictions
        preds = self.forward(signals)
        # compute and log loss
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        return optimizer
      
class SpikeNetInstance(InstanceBaseClass):
    def __init__(self,n_channels,lr=1e-4,weight_decay=1e-4):
        model = SpikeNet(eeg_channels=n_channels)
        print('this model comes with A TON of random hardcoded stuff!')
        head = self._identity
        super().__init__(model,head,lr,weight_decay)
        
    #this implementation comes with an integrated head
    def _identity(self,x):
        return x