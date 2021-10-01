import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStoppin

class LightningCatDogClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
    
    def __build_model(self):
        """Define model layers"""
        # Pretrained VGG16
        use_pretrained = True
        self.net = models.vgg16(pretrained=use_pretrained)
        # Change Output Size of Last FC Layer (4096 -> 1)
        self.net.classifier[6] = nn.Linear(in_features=self.net.classifier[6].in_features, out_features=2)
        # Specify The Layers for updating
        params_to_update = []
        update_params_name = ['classifier.6.weight', 'classifier.6.bias']

        for name, param in self.net.named_parameters():
            if name in update_params_name:
                param.requires_grad = True
                params_to_update.append(param)
            else:
                param.requires_grad = False
        
    def forward(self, x):
        return self.net(x)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        preds = self.forward(x)
        loss = self.cross_entropy_loss(preds, y)
        # self.log('train_loss', loss)
        # Calc Correct
        _, preds = torch.max(preds, 1)
        correct = torch.sum(preds == y).float() / preds.size(0)
        
        logs = {'train_loss': loss, 'train_correct': correct}
        
        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        preds = self.forward(x)
        loss = self.cross_entropy_loss(preds, y)
        # Calc Correct
        _, preds = torch.max(preds, 1)
        correct = torch.sum(preds == y).float() / preds.size(0)
        
        logs = {'val_loss': loss, 'val_correct': correct}
        
        return {'val_loss': loss, 'val_correct': correct, 'log': logs, 'progress_bar': logs}

    # Aggegate Validation Result
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_correct = torch.stack([x['val_correct'] for x in outputs]).mean()
        logs = {'avg_val_loss': avg_loss, 'avg_val_correct': avg_correct}
        torch.cuda.empty_cache()

        return {'avg_val_loss': avg_loss, 'log': logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class CatsDogsDataModule(pl.LightningDataModule):

    def setup(self, stage):
        # transforms for images
        transform=transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))])
        
        # prepare transforms standard to MNIST
        self.mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        self.mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)

        pathname = os.path.dirname(sys.argv[0])
        path = os.path.abspath(pathname)

        # dimensions of our images.
        img_width, img_height = 150, 150

        top_model_weights_path = 'model.h5'
        train_data_dir = os.path.join('data', 'train')
        validation_data_dir = os.path.join('data', 'validation')
        cats_train_path = os.path.join(path, train_data_dir, 'cats')
        nb_train_samples = 2 * len([name for name in os.listdir(cats_train_path)
                                    if os.path.isfile(
                                        os.path.join(cats_train_path, name))])
        nb_validation_samples = 800

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=64)

    def val_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=64)

data_module = CatsDogsDataModule()

# train
model = LightningCatDogClassifier()
trainer = pl.Trainer()

trainer.fit(model, data_module)