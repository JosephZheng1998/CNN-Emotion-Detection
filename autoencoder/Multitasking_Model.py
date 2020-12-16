import os
import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score
from IPython.display import Image
from IPython.core.display import Image, display
from torchvision.utils import save_image
import time
import pandas as pd
import matplotlib.pyplot as plt
import random

class EncoderBlock(nn.Module):
  def __init__(self, input_chan, output_chan):
    super(EncoderBlock, self).__init__()
    self.Conv1 = nn.Conv2d(in_channels=input_chan, out_channels=output_chan, kernel_size=(3,3),stride=(1,1), padding=(1,1))
    self.norm1 = nn.BatchNorm2d(output_chan)
    self.act1 = nn.ReLU()
    self.Conv2 = nn.Conv2d(in_channels=output_chan, out_channels=output_chan, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.norm2 = nn.BatchNorm2d(output_chan)
    self.act2 = nn.ReLU()
    self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2), return_indices=True)
  def forward(self,x):
    x=self.Conv1(x)
    x=self.norm1(x)
    x=self.act1(x)
    x=self.Conv2(x)
    x=self.norm2(x)
    x=self.act2(x)
    x,indices = self.pool(x)
    return (x, indices)



class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.encoderLayer1 = EncoderBlock(3,32)
    self.encoderLayer2 = EncoderBlock(32,64)
    self.encoderLayer3 = EncoderBlock(64,128)
    self.encoderLayer4 = EncoderBlock(128,256)
    self.encoderLin = nn.Sequential(
        nn.Flatten(),
        #change output to 686
        nn.Linear(256*3*3, 686)
    )
  def forward(self,x):
    indicesList = []
    x,indices1 = self.encoderLayer1(x)
    indicesList.append(indices1)
    x,indices2 = self.encoderLayer2(x)
    indicesList.append(indices2)
    x,indices3 = self.encoderLayer3(x)
    indicesList.append(indices3)
    x,indices4 = self.encoderLayer4(x)
    indicesList.append(indices4)
    x=self.encoderLin(x)
    return x, indicesList

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.TConv1 = nn.ConvTranspose2d(in_channels=256, out_channels=256,kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.norm1 = nn.BatchNorm2d(256)
    self.act1 = nn.ReLU()
    self.pool1 = nn.MaxUnpool2d(kernel_size=(2,2),stride=(2,2))
    self.TConv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128,kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.norm2 = nn.BatchNorm2d(128)
    self.act2 = nn.ReLU()

    self.pool2 = nn.MaxUnpool2d(kernel_size=(2,2),stride=(2,2))
    self.TConv3 = nn.ConvTranspose2d(in_channels=128, out_channels=128,kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.norm3 = nn.BatchNorm2d(128)
    self.act3 = nn.ReLU()
    self.TConv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64,kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.norm4 = nn.BatchNorm2d(64)
    self.act4 = nn.ReLU()

    self.pool3 = nn.MaxUnpool2d(kernel_size=(2,2),stride=(2,2))
    self.TConv5 = nn.ConvTranspose2d(in_channels=64, out_channels=64,kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.norm5 = nn.BatchNorm2d(64)
    self.act5 = nn.ReLU()
    self.TConv6 = nn.ConvTranspose2d(in_channels=64, out_channels=32,kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.norm6 = nn.BatchNorm2d(32)
    self.act6 = nn.ReLU()

    self.pool4 = nn.MaxUnpool2d(kernel_size=(2,2),stride=(2,2))
    self.TConv7 = nn.ConvTranspose2d(in_channels=32, out_channels=32,kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.norm7 = nn.BatchNorm2d(32)
    self.act7 = nn.ReLU()
    self.TConv8 = nn.ConvTranspose2d(in_channels=32, out_channels=3,kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.decoderLin = nn.Sequential(
        #change input to 686
        nn.Linear(686, 256*3*3)
    )
  def forward(self,x,indicesList):
    x=self.decoderLin(x)
    x=x.view(-1,256,3,3)
    x=self.TConv1(x)
    x=self.norm1(x)
    x=self.act1(x)
    x=self.pool1(x, indicesList[3])
    x=self.TConv2(x)
    x=self.norm2(x)
    x=self.act2(x)

    x=self.pool2(x, indicesList[2])
    x=self.TConv3(x)
    x=self.norm3(x)
    x=self.act3(x)
    x=self.TConv4(x)
    x=self.norm4(x)
    x=self.act4(x)

    x=self.pool3(x, indicesList[1])
    x=self.TConv5(x)
    x=self.norm5(x)
    x=self.act5(x)
    x=self.TConv6(x)
    x=self.norm6(x)
    x=self.act6(x)

    x=self.pool4(x, indicesList[0])
    x=self.TConv7(x)
    x=self.norm7(x)
    x=self.act7(x)
    x=self.TConv8(x)
    return x

class LinClassifier(nn.Module):
  def __init__(self):
    super(LinClassifier, self).__init__()
    self.layers = nn.Sequential(
        nn.ReLU(),
        #change input to 686
        nn.Linear(686,512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512,7)
    )
  def forward(self, x):
    return self.layers(x)


class AutoClassifier(nn.Module):
  def __init__(self):
    super(AutoClassifier, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()
    self.classifier = LinClassifier()
  def forwardAuto(self,x):
    x,indicesList=self.encoder(x)
    reconstructed=self.classifier(x, indicesList)
    return x
  def forwardClassify(self,x):
    x,indicesList = self.encoder(x)
    output = self.classifier(x)
    return x
