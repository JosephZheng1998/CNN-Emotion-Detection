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

def train(model, train_loader, optimizer, criterion, scaler, device, epoch, empty_cache=False):
    start = time.time()
    model.train()
    avg_loss = 0.0
    train_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(train_loader):
        feats, labels = feats.to(device), labels.to(device)
        
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(feats.float())
            loss = criterion(outputs, labels.long())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        avg_loss += loss.item()

        pred_labels = torch.max(F.softmax(outputs, dim=1), 1)[1]
        pred_labels = pred_labels.view(-1)
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        train_loss.extend([loss.item()]*feats.size()[0])

        if batch_num % 50 == 49:
            print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}\tElapsed Time: {:.4f}'.
                  format(epoch+1, batch_num+1, avg_loss/50, time.time()-start))
            avg_loss = 0.0    
        
        if empty_cache:
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss
        
    end = time.time()
    print('Epoch: {}\tTraining Time: {:.4f}'.format(epoch+1, end-start))
    return np.mean(train_loss), accuracy/total

def valid(model, val_loader, criterion, device, epoch, dataset, empty_cache=False):
    start = time.time()
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0
    pred_list = []
    true_list = []

    with torch.no_grad():
        for batch_num, (feats, labels) in enumerate(val_loader):
            feats, labels = feats.to(device), labels.to(device)
            outputs = model(feats.float())
            
            pred_labels = torch.max(F.softmax(outputs, dim=1), 1)[1]
            pred_labels = pred_labels.view(-1)
            
            loss = criterion(outputs, labels.long())
            
            accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            test_loss.extend([loss.item()]*feats.size()[0])

            pred_list.extend(pred_labels.detach().cpu().numpy())
            true_list.extend(labels.detach().cpu().numpy())

            if empty_cache:
                torch.cuda.empty_cache()
                del feats
                del labels
                del loss
    
    matrix = confusion_matrix(true_list, pred_list)
    model.train()
    end = time.time()
    print('Epoch: {}\t{} Validation Time: {:.4f}'.format(epoch+1, dataset, end-start))
    return np.mean(test_loss), accuracy/total, matrix

def train_model(model, model_name, train_loader, val_loader, test_loader, optimizer, criterion, scheduler, scaler, device, start_epoch, num_epochs, train_losses, train_accuracy, valid_losses, valid_accuracy, test_losses, test_accuracy, empty_cache=False):
    for epoch in range(start_epoch, num_epochs):
        print("Epoch: {}\tLearning Rate: {}".format(epoch+1, optimizer.param_groups[0]['lr']))
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, scaler, device, epoch, empty_cache)
        print('Epoch: {}\tTrain Loss: {:.5f}\tTrain Accuracy: {:.5f}'.format(epoch+1, train_loss, train_acc))
        train_losses.append(train_loss)
        train_accuracy.append(train_acc)
        
        val_loss, val_acc, matrix = valid(model, val_loader, criterion, device, epoch, 'Validation Dataset', empty_cache)
        print('Epoch: {}\tVal Loss: {:.5f}\tVal Accuracy: {:.5f}'.format(epoch+1, val_loss, val_acc))
        valid_losses.append(val_loss)
        valid_accuracy.append(val_acc)

        test_loss, test_acc, matrix = valid(model, test_loader, criterion, device, epoch, 'Test Dataset', empty_cache)
        print('Epoch: {}\tTest Loss: {:.5f}\tTest Accuracy: {:.5f}'.format(epoch+1, test_loss, test_acc))
        test_losses.append(test_loss)
        test_accuracy.append(test_acc)
        
        scheduler.step(val_loss)

        print('Epoch: {}\tSaved Model'.format(epoch+1))
        saved_model = {
            'train_epochs': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'criterion': criterion.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'train_losses': train_losses,
            'train_accuracy': train_accuracy,
            'valid_losses': valid_losses,
            'valid_accuracy': valid_accuracy,
            'test_losses': test_losses,
            'test_accuracy': test_accuracy
            }
        torch.save(saved_model, '{}.model'.format(model_name))    
    return train_losses, train_accuracy, valid_losses, valid_accuracy, test_losses, test_accuracy

def training_plot(a, b, c, name):
    plt.figure(1)
    plt.plot(a, 'b', label='train')
    plt.plot(b, 'g', label='valid')
    plt.plot(c, 'r', label='test')
    plt.title('Train/Valid/Test {}'.format(name))
    plt.legend()
    plt.show()
