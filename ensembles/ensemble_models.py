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

from plotting import make_plots, plot_matrix

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

def valid_average_ensemble(model_list, model_weights, val_loader, criterion, device, dataset, empty_cache=False):
    start = time.time()
    test_loss = []
    accuracy = 0
    total = 0
    true_label = []
    pred_label = []

    with torch.no_grad():
        for batch_num, (feats, labels) in enumerate(val_loader):
            feats, labels = feats.to(device), labels.to(device)
            output_list = []
            for name, model in model_list.items():
                model.eval()
                outputs = model(feats.float())
                output_list.append(outputs * model_weights[name])
            output_list = torch.stack(output_list)
            output_list = torch.mean(output_list, dim=0)
            
            pred_labels = torch.max(F.softmax(output_list, dim=1), 1)[1]
            pred_labels = pred_labels.view(-1)
            
            loss = criterion(output_list, labels.long())
            
            accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            test_loss.extend([loss.item()]*feats.size()[0])

            pred_label.extend(pred_labels.detach().cpu().numpy())
            true_label.extend(labels.detach().cpu().numpy())

            if empty_cache:
                torch.cuda.empty_cache()
                del feats
                del labels
                del loss
    
    matrix = confusion_matrix(true_label, pred_label)
    end = time.time()
    print('{} Validation Time: {:.4f}'.format(dataset, end-start))
    return np.mean(test_loss), accuracy/total, matrix

def valid_mode_ensemble(model_list, val_loader, criterion, device, dataset, empty_cache=False):
    start = time.time()
    test_loss = []
    accuracy = 0
    total = 0
    true_label = []
    pred_label = []

    with torch.no_grad():
        for batch_num, (feats, labels) in enumerate(val_loader):
            feats, labels = feats.to(device), labels.to(device)
            pred_list = []
            avg_loss = []
            for name, model in model_list.items():
                model.eval()
                outputs = model(feats.float())
                loss = criterion(outputs, labels.long())
                avg_loss.append(loss.item())
                pred_labels = torch.max(F.softmax(outputs, dim=1), 1)[1]
                pred_labels = pred_labels.view(-1)
                pred_list.append(pred_labels)
            
            avg_loss = np.mean(avg_loss)

            pred_list = torch.stack(pred_list)
            pred_list = torch.mode(pred_list, dim=0)[0]

            pred_label.extend(pred_list.detach().cpu().numpy())
            true_label.extend(labels.detach().cpu().numpy())
            
            accuracy += torch.sum(torch.eq(pred_list, labels)).item()
            total += len(labels)
            test_loss.extend([avg_loss]*feats.size()[0])

            if empty_cache:
                torch.cuda.empty_cache()
                del feats
                del labels
                del loss
    
    
    matrix = confusion_matrix(true_label, pred_label)
    end = time.time()
    print('{} Validation Time: {:.4f}'.format(dataset, end-start))
    return np.mean(test_loss), accuracy/total, matrix

def plot_ensembles(a, b):
    plt.figure(1)
    plt.plot(range(1, len(a) + 1), a, 'b', label='valid')
    plt.plot(range(1, len(b) + 1), b, 'g', label='test')
    plt.title('Ensemble Model Validation/Test Accuracy')
    plt.xlabel('Number of Models')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('ensemble_accuracy.png')
    plt.show()

if __name__ == '__main__':
    # run only once to set up validation and test folders
    splitfolders.ratio('test', output='val_test', seed=1337, ratio=(0, 0.5, 0.5), group_prefix=None)

    # check for GPU
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(device)
    
    train_loader, dev_loader, test_loader = preprocessing()
    
    # load individual models
    num_classes = 7

    train_accuracy = {}
    valid_accuracy = {}
    test_accuracy = {}
    model_list = {}

    criterion = nn.CrossEntropyLoss()
    empty_cache = True

    model_name = 'ResNet18'
    resnet18 = models.resnet18()
    resnet18.fc = nn.Linear(512, num_classes)
    if os.path.exists('{}.model'.format(model_name)):
        print('Found pretrained model!')
        saved_model = torch.load('{}.model'.format(model_name))
        resnet18.load_state_dict(saved_model['model'])
        model_list[model_name] = resnet18
        train_loss = saved_model['train_losses']
        valid_loss = saved_model['valid_losses']
        test_loss = saved_model['test_losses']
        train_accuracy[model_name] = saved_model['train_accuracy']
        valid_accuracy[model_name] = saved_model['valid_accuracy']
        test_accuracy[model_name] = saved_model['test_accuracy']
    resnet18.to(device)
    resnet18.eval()
    print('{} Test Loss: {}'.format(model_name, test_loss[-1]))
    make_plots(train_accuracy[model_name], valid_accuracy[model_name], model_name, 'Accuracy')
    make_plots(train_loss, valid_loss, model_name, 'Loss')
    loss, acc, matrix = valid(resnet18, dev_loader, criterion, device, 0, 'Validation Dataset', empty_cache)
    plot_matrix(matrix, 'Validation', model_name)
    loss, acc, matrix = valid(resnet18, test_loader, criterion, device, 0, 'Test Dataset', empty_cache)
    plot_matrix(matrix, 'Test', model_name)

    model_name = 'ResNet34'
    resnet34 = models.resnet34()
    resnet34.fc = nn.Linear(512, num_classes)
    if os.path.exists('{}.model'.format(model_name)):
        print('Found pretrained model!')
        saved_model = torch.load('{}.model'.format(model_name))
        resnet34.load_state_dict(saved_model['model'])
        model_list[model_name] = resnet34
        train_loss = saved_model['train_losses']
        valid_loss = saved_model['valid_losses']
        test_loss = saved_model['test_losses']
        train_accuracy[model_name] = saved_model['train_accuracy']
        valid_accuracy[model_name] = saved_model['valid_accuracy']
        test_accuracy[model_name] = saved_model['test_accuracy']
    resnet34.to(device)
    resnet34.eval()
    print('{} Test Loss: {}'.format(model_name, test_loss[-1]))
    make_plots(train_accuracy[model_name], valid_accuracy[model_name], model_name, 'Accuracy')
    make_plots(train_loss, valid_loss, model_name, 'Loss')
    loss, acc, matrix = valid(resnet34, dev_loader, criterion, device, 0, 'Validation Dataset', empty_cache)
    plot_matrix(matrix, 'Validation', model_name)
    loss, acc, matrix = valid(resnet34, test_loader, criterion, device, 0, 'Test Dataset', empty_cache)
    plot_matrix(matrix, 'Test', model_name)

    model_name = 'ResNet50'
    resnet50 = models.resnet50()
    resnet50.fc = nn.Linear(2048, num_classes)
    if os.path.exists('{}.model'.format(model_name)):
        print('Found pretrained model!')
        saved_model = torch.load('{}.model'.format(model_name))
        resnet50.load_state_dict(saved_model['model'])
        model_list[model_name] = resnet50
        train_loss = saved_model['train_losses']
        valid_loss = saved_model['valid_losses']
        test_loss = saved_model['test_losses']
        train_accuracy[model_name] = saved_model['train_accuracy']
        valid_accuracy[model_name] = saved_model['valid_accuracy']
        test_accuracy[model_name] = saved_model['test_accuracy']
    resnet50.to(device)
    resnet50.eval()
    print('{} Test Loss: {}'.format(model_name, test_loss[-1]))
    make_plots(train_accuracy[model_name], valid_accuracy[model_name], model_name, 'Accuracy')
    make_plots(train_loss, valid_loss, model_name, 'Loss')
    loss, acc, matrix = valid(resnet50, dev_loader, criterion, device, 0, 'Validation Dataset', empty_cache)
    plot_matrix(matrix, 'Validation', model_name)
    loss, acc, matrix = valid(resnet50, test_loader, criterion, device, 0, 'Test Dataset', empty_cache)
    plot_matrix(matrix, 'Test', model_name)

    model_name = 'ResNet101'
    resnet101 = models.resnet101()
    resnet101.fc = nn.Linear(2048, num_classes)
    if os.path.exists('{}.model'.format(model_name)):
        print('Found pretrained model!')
        saved_model = torch.load('{}.model'.format(model_name))
        resnet101.load_state_dict(saved_model['model'])
        model_list[model_name] = resnet101
        train_loss = saved_model['train_losses']
        valid_loss = saved_model['valid_losses']
        test_loss = saved_model['test_losses']
        train_accuracy[model_name] = saved_model['train_accuracy']
        valid_accuracy[model_name] = saved_model['valid_accuracy']
        test_accuracy[model_name] = saved_model['test_accuracy']
    resnet101.to(device)
    resnet101.eval()
    print('{} Test Loss: {}'.format(model_name, test_loss[-1]))
    make_plots(train_accuracy[model_name], valid_accuracy[model_name], model_name, 'Accuracy')
    make_plots(train_loss, valid_loss, model_name, 'Loss')
    loss, acc, matrix = valid(resnet101, dev_loader, criterion, device, 0, 'Validation Dataset', empty_cache)
    plot_matrix(matrix, 'Validation', model_name)
    loss, acc, matrix = valid(resnet101, test_loader, criterion, device, 0, 'Test Dataset', empty_cache)
    plot_matrix(matrix, 'Test', model_name)

    model_name = 'ResNet152'
    resnet152 = models.resnet152()
    resnet152.fc = nn.Linear(2048, num_classes)
    if os.path.exists('{}.model'.format(model_name)):
        print('Found pretrained model!')
        saved_model = torch.load('{}.model'.format(model_name))
        resnet152.load_state_dict(saved_model['model'])
        model_list[model_name] = resnet152
        train_loss = saved_model['train_losses']
        valid_loss = saved_model['valid_losses']
        test_loss = saved_model['test_losses']
        train_accuracy[model_name] = saved_model['train_accuracy']
        valid_accuracy[model_name] = saved_model['valid_accuracy']
        test_accuracy[model_name] = saved_model['test_accuracy']
    resnet152.to(device)
    resnet152.eval()
    print('{} Test Loss: {}'.format(model_name, test_loss[-1]))
    make_plots(train_accuracy[model_name], valid_accuracy[model_name], model_name, 'Accuracy')
    make_plots(train_loss, valid_loss, model_name, 'Loss')
    loss, acc, matrix = valid(resnet152, dev_loader, criterion, device, 0, 'Validation Dataset', empty_cache)
    plot_matrix(matrix, 'Validation', model_name)
    loss, acc, matrix = valid(resnet152, test_loader, criterion, device, 0, 'Test Dataset', empty_cache)
    plot_matrix(matrix, 'Test', model_name)

    model_name = 'ResNeXt50'
    resnext50 = models.resnext50_32x4d()
    resnext50.fc = nn.Linear(2048, num_classes)
    if os.path.exists('{}.model'.format(model_name)):
        print('Found pretrained model!')
        saved_model = torch.load('{}.model'.format(model_name))
        resnext50.load_state_dict(saved_model['model'])
        model_list[model_name] = resnext50
        train_loss = saved_model['train_losses']
        valid_loss = saved_model['valid_losses']
        test_loss = saved_model['test_losses']
        train_accuracy[model_name] = saved_model['train_accuracy']
        valid_accuracy[model_name] = saved_model['valid_accuracy']
        test_accuracy[model_name] = saved_model['test_accuracy']
    resnext50.to(device)
    resnext50.eval()
    print('{} Test Loss: {}'.format(model_name, test_loss[-1]))
    make_plots(train_accuracy[model_name], valid_accuracy[model_name], model_name, 'Accuracy')
    make_plots(train_loss, valid_loss, model_name, 'Loss')
    loss, acc, matrix = valid(resnext50, dev_loader, criterion, device, 0, 'Validation Dataset', empty_cache)
    plot_matrix(matrix, 'Validation', model_name)
    loss, acc, matrix = valid(resnext50, test_loader, criterion, device, 0, 'Test Dataset', empty_cache)
    plot_matrix(matrix, 'Test', model_name)

    model_name = 'ResNeXt101'
    resnext101 = models.resnext101_32x8d()
    resnext101.fc = nn.Linear(2048, num_classes)
    if os.path.exists('{}.model'.format(model_name)):
        print('Found pretrained model!')
        saved_model = torch.load('{}.model'.format(model_name))
        resnext101.load_state_dict(saved_model['model'])
        model_list[model_name] = resnext101
        train_loss = saved_model['train_losses']
        valid_loss = saved_model['valid_losses']
        test_loss = saved_model['test_losses']
        train_accuracy[model_name] = saved_model['train_accuracy']
        valid_accuracy[model_name] = saved_model['valid_accuracy']
        test_accuracy[model_name] = saved_model['test_accuracy']
    resnext101.to(device)
    resnext101.eval()
    print('{} Test Loss: {}'.format(model_name, test_loss[-1]))
    make_plots(train_accuracy[model_name], valid_accuracy[model_name], model_name, 'Accuracy')
    make_plots(train_loss, valid_loss, model_name, 'Loss')
    loss, acc, matrix = valid(resnext101, dev_loader, criterion, device, 0, 'Validation Dataset', empty_cache)
    plot_matrix(matrix, 'Validation', model_name)
    loss, acc, matrix = valid(resnext101, test_loader, criterion, device, 0, 'Test Dataset', empty_cache)
    plot_matrix(matrix, 'Test', model_name)

    model_name = 'Wide_ResNet50'
    wide_resnet50 = models.wide_resnet50_2()
    wide_resnet50.fc = nn.Linear(2048, num_classes)
    if os.path.exists('{}.model'.format(model_name)):
        print('Found pretrained model!')
        saved_model = torch.load('{}.model'.format(model_name))
        wide_resnet50.load_state_dict(saved_model['model'])
        model_list[model_name] = wide_resnet50
        train_loss = saved_model['train_losses']
        valid_loss = saved_model['valid_losses']
        test_loss = saved_model['test_losses']
        train_accuracy[model_name] = saved_model['train_accuracy']
        valid_accuracy[model_name] = saved_model['valid_accuracy']
        test_accuracy[model_name] = saved_model['test_accuracy']
    wide_resnet50.to(device)
    wide_resnet50.eval()
    print('{} Test Loss: {}'.format(model_name, test_loss[-1]))
    make_plots(train_accuracy[model_name], valid_accuracy[model_name], model_name, 'Accuracy')
    make_plots(train_loss, valid_loss, model_name, 'Loss')
    loss, acc, matrix = valid(wide_resnet50, dev_loader, criterion, device, 0, 'Validation Dataset', empty_cache)
    plot_matrix(matrix, 'Validation', model_name)
    loss, acc, matrix = valid(wide_resnet50, test_loader, criterion, device, 0, 'Test Dataset', empty_cache)
    plot_matrix(matrix, 'Test', model_name)

    model_name = 'Wide_ResNet101'
    wide_resnet101 = models.wide_resnet101_2()
    wide_resnet101.fc = nn.Linear(2048, num_classes)
    if os.path.exists('{}.model'.format(model_name)):
        print('Found pretrained model!')
        saved_model = torch.load('{}.model'.format(model_name))
        wide_resnet101.load_state_dict(saved_model['model'])
        model_list[model_name] = wide_resnet101
        train_loss = saved_model['train_losses']
        valid_loss = saved_model['valid_losses']
        test_loss = saved_model['test_losses']
        train_accuracy[model_name] = saved_model['train_accuracy']
        valid_accuracy[model_name] = saved_model['valid_accuracy']
        test_accuracy[model_name] = saved_model['test_accuracy']
    wide_resnet101.to(device)
    wide_resnet101.eval()
    print('{} Test Loss: {}'.format(model_name, test_loss[-1]))
    make_plots(train_accuracy[model_name], valid_accuracy[model_name], model_name, 'Accuracy')
    make_plots(train_loss, valid_loss, model_name, 'Loss')
    loss, acc, matrix = valid(wide_resnet101, dev_loader, criterion, device, 0, 'Validation Dataset', empty_cache)
    plot_matrix(matrix, 'Validation', model_name)
    loss, acc, matrix = valid(wide_resnet101, test_loader, criterion, device, 0, 'Test Dataset', empty_cache)
    plot_matrix(matrix, 'Test', model_name)

    model_name = 'DenseNet121'
    densenet121 = models.densenet121()
    densenet121.classifier = nn.Linear(1024, num_classes)
    if os.path.exists('{}.model'.format(model_name)):
        print('Found pretrained model!')
        saved_model = torch.load('{}.model'.format(model_name))
        densenet121.load_state_dict(saved_model['model'])
        model_list[model_name] = densenet121
        train_loss = saved_model['train_losses']
        valid_loss = saved_model['valid_losses']
        test_loss = saved_model['test_losses']
        train_accuracy[model_name] = saved_model['train_accuracy']
        valid_accuracy[model_name] = saved_model['valid_accuracy']
        test_accuracy[model_name] = saved_model['test_accuracy']
    densenet121.to(device)
    densenet121.eval()
    print('{} Test Loss: {}'.format(model_name, test_loss[-1]))
    make_plots(train_accuracy[model_name], valid_accuracy[model_name], model_name, 'Accuracy')
    make_plots(train_loss, valid_loss, model_name, 'Loss')
    loss, acc, matrix = valid(densenet121, dev_loader, criterion, device, 0, 'Validation Dataset', empty_cache)
    plot_matrix(matrix, 'Validation', model_name)
    loss, acc, matrix = valid(densenet121, test_loader, criterion, device, 0, 'Test Dataset', empty_cache)
    plot_matrix(matrix, 'Test', model_name)

    model_name = 'DenseNet161'
    densenet161 = models.densenet161()
    densenet161.classifier = nn.Linear(2208, num_classes)
    if os.path.exists('{}.model'.format(model_name)):
        print('Found pretrained model!')
        saved_model = torch.load('{}.model'.format(model_name))
        densenet161.load_state_dict(saved_model['model'])
        model_list[model_name] = densenet161
        train_loss = saved_model['train_losses']
        valid_loss = saved_model['valid_losses']
        test_loss = saved_model['test_losses']
        train_accuracy[model_name] = saved_model['train_accuracy']
        valid_accuracy[model_name] = saved_model['valid_accuracy']
        test_accuracy[model_name] = saved_model['test_accuracy']
    densenet161.to(device)
    densenet161.eval()
    print('{} Test Loss: {}'.format(model_name, test_loss[-1]))
    make_plots(train_accuracy[model_name], valid_accuracy[model_name], model_name, 'Accuracy')
    make_plots(train_loss, valid_loss, model_name, 'Loss')
    loss, acc, matrix = valid(densenet161, dev_loader, criterion, device, 0, 'Validation Dataset', empty_cache)
    plot_matrix(matrix, 'Validation', model_name)
    loss, acc, matrix = valid(densenet161, test_loader, criterion, device, 0, 'Test Dataset', empty_cache)
    plot_matrix(matrix, 'Test', model_name)

    model_name = 'DenseNet169'
    densenet169 = models.densenet169()
    densenet169.classifier = nn.Linear(1664, num_classes)
    if os.path.exists('{}.model'.format(model_name)):
        print('Found pretrained model!')
        saved_model = torch.load('{}.model'.format(model_name))
        densenet169.load_state_dict(saved_model['model'])
        model_list[model_name] = densenet169
        train_loss = saved_model['train_losses']
        valid_loss = saved_model['valid_losses']
        test_loss = saved_model['test_losses']
        train_accuracy[model_name] = saved_model['train_accuracy']
        valid_accuracy[model_name] = saved_model['valid_accuracy']
        test_accuracy[model_name] = saved_model['test_accuracy']
    densenet169.to(device)
    densenet169.eval()
    print('{} Test Loss: {}'.format(model_name, test_loss[-1]))
    make_plots(train_accuracy[model_name], valid_accuracy[model_name], model_name, 'Accuracy')
    make_plots(train_loss, valid_loss, model_name, 'Loss')
    loss, acc, matrix = valid(densenet169, dev_loader, criterion, device, 0, 'Validation Dataset', empty_cache)
    plot_matrix(matrix, 'Validation', model_name)
    loss, acc, matrix = valid(densenet169, test_loader, criterion, device, 0, 'Test Dataset', empty_cache)
    plot_matrix(matrix, 'Test', model_name)

    model_name = 'DenseNet201'
    densenet201 = models.densenet201()
    densenet201.classifier = nn.Linear(1920, num_classes)
    if os.path.exists('{}.model'.format(model_name)):
        print('Found pretrained model!')
        saved_model = torch.load('{}.model'.format(model_name))
        densenet201.load_state_dict(saved_model['model'])
        model_list[model_name] = densenet201
        train_loss = saved_model['train_losses']
        valid_loss = saved_model['valid_losses']
        test_loss = saved_model['test_losses']
        train_accuracy[model_name] = saved_model['train_accuracy']
        valid_accuracy[model_name] = saved_model['valid_accuracy']
        test_accuracy[model_name] = saved_model['test_accuracy']
    densenet201.to(device)
    densenet201.eval()
    print('{} Test Loss: {}'.format(model_name, test_loss[-1]))
    make_plots(train_accuracy[model_name], valid_accuracy[model_name], model_name, 'Accuracy')
    make_plots(train_loss, valid_loss, model_name, 'Loss')
    loss, acc, matrix = valid(densenet201, dev_loader, criterion, device, 0, 'Validation Dataset', empty_cache)
    plot_matrix(matrix, 'Validation', model_name)
    loss, acc, matrix = valid(densenet201, test_loader, criterion, device, 0, 'Test Dataset', empty_cache)
    plot_matrix(matrix, 'Test', model_name)

    model_name = 'VGG11'
    vgg11 = models.vgg11_bn()
    vgg11.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, num_classes)
    )
    if os.path.exists('{}.model'.format(model_name)):
        print('Found pretrained model!')
        saved_model = torch.load('{}.model'.format(model_name))
        vgg11.load_state_dict(saved_model['model'])
        model_list[model_name] = vgg11
        train_loss = saved_model['train_losses']
        valid_loss = saved_model['valid_losses']
        test_loss = saved_model['test_losses']
        train_accuracy[model_name] = saved_model['train_accuracy']
        valid_accuracy[model_name] = saved_model['valid_accuracy']
        test_accuracy[model_name] = saved_model['test_accuracy']
    vgg11.to(device)
    vgg11.eval()
    print('{} Test Loss: {}'.format(model_name, test_loss[-1]))
    make_plots(train_accuracy[model_name], valid_accuracy[model_name], model_name, 'Accuracy')
    make_plots(train_loss, valid_loss, model_name, 'Loss')
    loss, acc, matrix = valid(vgg11, dev_loader, criterion, device, 0, 'Validation Dataset', empty_cache)
    plot_matrix(matrix, 'Validation', model_name)
    loss, acc, matrix = valid(vgg11, test_loader, criterion, device, 0, 'Test Dataset', empty_cache)
    plot_matrix(matrix, 'Test', model_name)

    model_name = 'VGG13'
    vgg13 = models.vgg13_bn()
    vgg13.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, num_classes)
    )
    if os.path.exists('{}.model'.format(model_name)):
        print('Found pretrained model!')
        saved_model = torch.load('{}.model'.format(model_name))
        vgg13.load_state_dict(saved_model['model'])
        model_list[model_name] = vgg13
        train_loss = saved_model['train_losses']
        valid_loss = saved_model['valid_losses']
        test_loss = saved_model['test_losses']
        train_accuracy[model_name] = saved_model['train_accuracy']
        valid_accuracy[model_name] = saved_model['valid_accuracy']
        test_accuracy[model_name] = saved_model['test_accuracy']
    vgg13.to(device)
    vgg13.eval()
    print('{} Test Loss: {}'.format(model_name, test_loss[-1]))
    make_plots(train_accuracy[model_name], valid_accuracy[model_name], model_name, 'Accuracy')
    make_plots(train_loss, valid_loss, model_name, 'Loss')
    loss, acc, matrix = valid(vgg13, dev_loader, criterion, device, 0, 'Validation Dataset', empty_cache)
    plot_matrix(matrix, 'Validation', model_name)
    loss, acc, matrix = valid(vgg13, test_loader, criterion, device, 0, 'Test Dataset', empty_cache)
    plot_matrix(matrix, 'Test', model_name)

    model_name = 'VGG16'
    vgg16 = models.vgg16_bn()
    vgg16.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, num_classes)
    )
    if os.path.exists('{}.model'.format(model_name)):
        print('Found pretrained model!')
        saved_model = torch.load('{}.model'.format(model_name))
        vgg16.load_state_dict(saved_model['model'])
        model_list[model_name] = vgg16
        train_loss = saved_model['train_losses']
        valid_loss = saved_model['valid_losses']
        test_loss = saved_model['test_losses']
        train_accuracy[model_name] = saved_model['train_accuracy']
        valid_accuracy[model_name] = saved_model['valid_accuracy']
        test_accuracy[model_name] = saved_model['test_accuracy']
    vgg16.to(device)
    vgg16.eval()
    print('{} Test Loss: {}'.format(model_name, test_loss[-1]))
    make_plots(train_accuracy[model_name], valid_accuracy[model_name], model_name, 'Accuracy')
    make_plots(train_loss, valid_loss, model_name, 'Loss')
    loss, acc, matrix = valid(vgg16, dev_loader, criterion, device, 0, 'Validation Dataset', empty_cache)
    plot_matrix(matrix, 'Validation', model_name)
    loss, acc, matrix = valid(vgg16, test_loader, criterion, device, 0, 'Test Dataset', empty_cache)
    plot_matrix(matrix, 'Test', model_name)

    model_name = 'VGG19'
    vgg19 = models.vgg19_bn()
    vgg19.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, num_classes)
    )
    if os.path.exists('{}.model'.format(model_name)):
        print('Found pretrained model!')
        saved_model = torch.load('{}.model'.format(model_name))
        vgg19.load_state_dict(saved_model['model'])
        model_list[model_name] = vgg19
        train_loss = saved_model['train_losses']
        valid_loss = saved_model['valid_losses']
        test_loss = saved_model['test_losses']
        train_accuracy[model_name] = saved_model['train_accuracy']
        valid_accuracy[model_name] = saved_model['valid_accuracy']
        test_accuracy[model_name] = saved_model['test_accuracy']
    vgg19.to(device)
    vgg19.eval()
    print('{} Test Loss: {}'.format(model_name, test_loss[-1]))
    make_plots(train_accuracy[model_name], valid_accuracy[model_name], model_name, 'Accuracy')
    make_plots(train_loss, valid_loss, model_name, 'Loss')
    loss, acc, matrix = valid(vgg19, dev_loader, criterion, device, 0, 'Validation Dataset', empty_cache)
    plot_matrix(matrix, 'Validation', model_name)
    loss, acc, matrix = valid(vgg19, test_loader, criterion, device, 0, 'Test Dataset', empty_cache)
    plot_matrix(matrix, 'Test', model_name)

    test_df = pd.DataFrame(test_accuracy, index=range(1, 31))
    test_df.to_csv('model_accuracy.csv')

    plt.figure(figsize=(12,8))
    ax = sns.lineplot(data=test_df)
    ax.set(xlabel='Epochs', ylabel='Accuracy', title="Test Accuracy of 17 CNN Models")
    plt.savefig('model_accuracy.png')
    plt.show()

    # sort models based on testing accuracy
    test_dict = dict(test_df.loc[30])
    sorted_models = [i[0] for i in sorted(test_dict.items(), key=lambda k:k[1], reverse=True)]

    # try all 17 model combinations
    criterion = nn.CrossEntropyLoss()
    empty_cache = True
    train_acc = []
    val_acc = []
    test_acc = []
    for i in range(1, 18):
        print('Evaluating Top {} Model Ensemble'.format(i))
        selected_models = sorted_models[:i]
        print('Selected Models:', selected_models)
        selected_model_list = {k:v for k,v in model_list.items() if k in selected_models}
        loss, acc, matrix = valid_mode_ensemble(selected_model_list, train_loader, criterion, device, 'Train Dataset', empty_cache)
        train_acc.append(acc)
        print('Train Loss: {:.5f}\tTrain Accuracy: {:.5f}'.format(loss, acc))
        plot_matrix(matrix, 'Train', 'Top {} Model Ensemble'.format(i))
        loss, acc, matrix = valid_mode_ensemble(selected_model_list, dev_loader, criterion, device, 'Validation Dataset', empty_cache)
        val_acc.append(acc)
        print('Validation Loss: {:.5f}\tValidation Accuracy: {:.5f}'.format(loss, acc))
        plot_matrix(matrix, 'Validation', 'Top {} Model Ensemble'.format(i))
        loss, acc, matrix = valid_mode_ensemble(selected_model_list, test_loader, criterion, device, 'Test Dataset', empty_cache)
        test_acc.append(acc)
        print('Test Loss: {:.5f}\tTest Accuracy: {:.5f}'.format(loss, acc))
        plot_matrix(matrix, 'Test', 'Top {} Model Ensemble'.format(i))

    plot_ensembles(val_acc, test_acc)

    acc = [train_acc, val_acc, test_acc]
    acc_df = pd.DataFrame(acc, index=['Training Accuracy', 'Validation Accuracy', 'Testing Accuracy'], columns=['Top {} Model Ensemble'.format(i) for i in range(1, 18)])
    acc_df.to_csv('ensemble_accuracy.csv')
