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
from sklearn.metrics import confusion_matrix
import seaborn as sns

import time
import pandas as pd
import matplotlib.pyplot as plt

import splitfolders

from training import train_model, training_plot

def preprocessing():
    batch_size = 64
    train_transform = transforms.Compose([transforms.Resize(224),
                                          transforms.RandomRotation(45),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_transform = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = ImageFolder(root='train/', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

    dev_dataset = ImageFolder(root='val_test/val/', transform=val_transform)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    test_dataset = ImageFolder(root='val_test/test/', transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    print('train dataset: {} images {} classes'.format(len(train_dataset), len(train_dataset.classes)))
    print('dev dataset: {} images {} classes'.format(len(dev_dataset), len(dev_dataset.classes)))
    print('test dataset: {} images {} classes'.format(len(test_dataset), len(test_dataset.classes)))

    return train_loader, dev_loader, test_loader

if __name__ == '__main__':
    # run only once to set up validation and test folders
    splitfolders.ratio('test', output='val_test', seed=1337, ratio=(0, 0.5, 0.5), group_prefix=None)

    # check for GPU
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(device)
    
    train_loader, dev_loader, test_loader = preprocessing()
    
    num_epochs = 30
    num_classes = len(train_dataset.classes)

    lr = 1e-2
    weight_decay = 5e-4
    momentum = 0.9

    start_epoch = 0
    train_losses = []
    train_accuracy = []
    valid_losses = []
    valid_accuracy = []
    test_losses = []
    test_accuracy = []

    empty_cache = True

    model_name = 'DenseNet121'
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(1024, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)
    scaler = torch.cuda.amp.GradScaler()

    if os.path.exists('{}.model'.format(model_name)):
        print('Found pretrained model!')
        saved_model = torch.load('{}.model'.format(model_name))
        start_epoch = saved_model['train_epochs']
        model.load_state_dict(saved_model['model'])
        criterion.load_state_dict(saved_model['criterion'])
        optimizer.load_state_dict(saved_model['optimizer'])
        scheduler.load_state_dict(saved_model['scheduler'])
        scaler.load_state_dict(saved_model['scaler'])
        train_losses = saved_model['train_losses']
        train_accuracy = saved_model['train_accuracy']
        valid_losses = saved_model['valid_losses']
        valid_accuracy = saved_model['valid_accuracy']
        test_losses = saved_model['test_losses']
        test_accuracy = saved_model['test_accuracy']

    print('Training', model_name)
    train_losses, train_accuracy, valid_losses, valid_accuracy, test_losses, test_accuracy = train_model(model, model_name, train_loader, dev_loader, test_loader, optimizer, criterion, scheduler, scaler, device, start_epoch, num_epochs, train_losses, train_accuracy, valid_losses, valid_accuracy, test_losses, test_accuracy, empty_cache)

    training_plot(train_losses, valid_losses, test_losses, 'Loss')

    training_plot(train_accuracy, valid_accuracy, test_accuracy, 'Accuracy')
