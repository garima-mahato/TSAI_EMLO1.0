import os
import math
import random
import numpy

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms as T, datasets, models
import pytorch_lightning as pl

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class CIFAR100Classifier(pl.LightningModule):
    def __init__(self, data_dir, batch_size=128, num_workers=4, num_classes=100, pretrained=True):
        super().__init__()
        # Set up class attributes
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.num_classes = num_classes
        
        ## Model
        # Use pretrained Resnet34 models
        self.network = models.resnet34(pretrained=pretrained)
        # Replace last layer
        self.network.fc = nn.Linear(self.network.fc.in_features, self.num_classes)
    
    def forward(self, x):
        return self.network(x)
    
    def prepare_data(self):
        means, stds = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transforms = {'train': T.Compose([T.ToTensor(), T.Normalize(means, stds, inplace=True)]), 'valid': T.Compose([T.ToTensor(), T.Normalize(means, stds, inplace=True)]), 'test': T.Compose([T.ToTensor(), T.Normalize(means, stds, inplace=True)])}
        
        dataset = {}
        dataset['train'] = datasets.CIFAR100(root=f'{self.data_dir}/cifar100_dataset/train', train=True, download=True, transform=transforms['train'])
        dataset['valid'] = datasets.CIFAR100(root=f'{self.data_dir}/cifar100_dataset/valid', train=False, download=True, transform=transforms['valid'])
            #datasets.ImageFolder(self.datadir+f'/{key.upper()}', transform=transforms[key])
        
        dataloader = {}
        for key in ['train','valid']:
            if key == 'train':
                dataloader[key] = DataLoader(dataset[key], self.batch_size, shuffle=True)
            else:
                dataloader[key] = DataLoader(dataset[key], self.batch_size)
        
        self.train_loader, self.val_loader = dataloader['train'], dataloader['valid']
        #return dataloader['train'], dataloader['valid'], dataloader['test']
    
    def train_dataloader(self):
        '''
        Returns:
        (torch.utils.data.DataLoader): Training set data loader
        '''
        return self.train_loader

    def val_dataloader(self):
        '''
        Returns:
        (torch.utils.data.DataLoader): Validation set data loader
        '''
        return self.val_loader
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
    
    def training_step(self, batch, batch_idx):
        # Get inputs and output from batch
        x, labels = batch
        
        # Compute prediction through the network
        prediction = self.forward(x)
        
        loss = F.cross_entropy(prediction, labels)
        
        ## Log training loss
        #logs = {'train_loss': loss.item()}
        
        output = {
            'loss': loss,
            'train_loss': loss.item()
        }
        
        return output
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        prediction = self.forward(x)  # Generate predictions
        loss = F.cross_entropy(prediction, labels)   # Calculate loss
        acc = accuracy(prediction, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}],{} train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, "last_lr: {:.5f},".format(result['lrs'][-1]) if 'lrs' in result else '', 
            result['train_loss'], result['val_loss'], result['val_acc']))
    