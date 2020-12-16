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

def preprocessing():
    batch_size = 64
    train_transform = transforms.Compose([
                                          transforms.RandomRotation(45),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_transform = transforms.Compose([
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

# train the multitasking model
def train(model, train_loader, val_loader, recon_criterion, class_criterion, optimizer, scheduler, epochs, device):
    t_start = time.time()
    train_recon_losses = []
    train_class_losses = []
    train_accs = []
    val_class_losses = []
    val_recon_losses = []
    val_accs = []
    for epoch in range(epochs):
        model.train()
        running_recon_loss = 0
        running_class_loss = 0
        total = 0
        correct = 0
        for idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            features, idx_list = model.encoder.forward(images)
            output = model.classifier.forward(features)
            recon_images = model.decoder.forward(features, idx_list)
            class_loss = class_criterion(output, labels)
            recon_loss = recon_criterion(recon_images, images)
            total_loss = class_loss + 10*recon_loss
            total_loss.backward()
            #class_loss.backward()
            optimizer.step()
            preds = torch.argmax(output, dim=1)
            correct += (preds == labels).sum().item()
            running_recon_loss += float(recon_loss.item() * images.shape[0])
            running_class_loss += float(class_loss.item() * images.shape[0])
            total += images.shape[0]
            #recon_loss = 0
            del images
            del labels
            del recon_images
            del features
            del idx_list
            del output
            del recon_loss
            del class_loss
            del preds
            del total_loss
            torch.cuda.empty_cache()
        val_class_loss, val_recon_loss, val_acc = validate(model, val_loader, recon_criterion, class_criterion, device)
        val_total_loss = val_recon_loss + val_class_loss
        train_recon_loss = running_recon_loss/total
        train_class_loss = running_class_loss/total
        train_total_loss = train_recon_loss + train_class_loss
        train_acc = correct/total
        train_recon_losses.append(train_recon_loss)
        train_class_losses.append(train_class_loss)
        train_accs.append(train_acc)
        val_class_losses.append(val_class_loss)
        val_recon_losses.append(val_recon_loss)
        val_accs.append(val_acc)
        scheduler.step(val_class_loss)
        to_print = "Epoch: {}/{}, Training Time:{:.2f}, Trained Samples: {}/{}, Train Total Loss: {:.5f}, Train Recon Loss: {:.5f}, Train Class Loss: {:.5f} Train Accuracy: {:.5f}, Val Total Loss: {:.5f}, Val Recon Loss: {:.5f}, Val Class Loss: {:.5f}, Val Accuracy: {:.5f}".format(
                epoch+1, epochs, time.time()-t_start, total, len(train_loader.dataset), train_total_loss, train_recon_loss, train_class_loss, train_acc,
                    val_total_loss, val_recon_loss, val_class_loss, val_acc)
        print(to_print)
        if (epoch+1) % 10 == 0:
            saved_model = {
                        'train_epochs': epoch + 1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'recon_criterion': recon_criterion.state_dict(),
                        'class_criterion': class_criterion.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'train_class_losses': train_class_losses,
                        'train_recon_losses': train_recon_losses,
                        'train_accuracy': train_accs,
                        'valid_class_losses': val_class_losses,
                        'valid_accuracy': val_accs,
                        'val_recon_losses': val_recon_losses,
                        }
            torch.save(saved_model, 'gdrive/MyDrive/complete_model{}'.format(epoch+1))
    return train_recon_losses, train_class_losses, train_accs, val_class_losses, val_recon_losses, val_accs

# validate the multitasking model
def validate(model, val_loader, recon_criterion, class_criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_recon_loss = 0
    running_class_loss = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            features, idx_list = model.encoder.forward(images)
            output = model.classifier.forward(features)
            recon_images = model.decoder.forward(features, idx_list)
            class_loss = class_criterion(output, labels)
            recon_loss = recon_criterion(recon_images, images)
            running_recon_loss += float(recon_loss.item() * labels.shape[0])
            running_class_loss += float(class_loss * labels.shape[0])
            total += labels.shape[0]
            preds = torch.argmax(output, dim=1)
            correct += (preds == labels).sum().item()
            """
            print(correct)
            print(preds)
            print(labels)
            """
            del images
            del labels
            del recon_images
            del features
            del idx_list
            del output
            del recon_loss
            del class_loss
            del preds
            torch.cuda.empty_cache()
    return running_class_loss/total, running_recon_loss/total, correct/total


# only train the autoencoder part of the model
def pretrain(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    t_start = time.time()
    train_recon_losses = []
    val_recon_losses = []
    for epoch in range(epochs):
        model.train()
        running_recon_loss = 0
        total = 0
        for idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            features, idx_list = model.encoder.forward(images)
            recon_images = model.decoder.forward(features, idx_list)
            recon_loss = criterion(recon_images, images)
            recon_loss.backward()
            optimizer.step()
            running_recon_loss += float(recon_loss.item() * images.shape[0])
            total += images.shape[0]
            del images
            del labels
            del recon_images
            del features
            del idx_list
            del recon_loss
            torch.cuda.empty_cache()
        val_recon_loss = pre_validate(model, val_loader, criterion, device)
        train_recon_loss = running_recon_loss/total
        train_recon_losses.append(train_recon_loss)
        val_recon_losses.append(val_recon_loss)
        scheduler.step(val_recon_loss)
        to_print = "Epoch: {}/{}, Training Time:{:.2f}, Trained Samples: {}/{}, Train Recon Loss: {:.5f}, Val Recon Loss: {:.5f}".format(
                epoch+1, epochs, time.time()-t_start, total, len(train_loader.dataset), train_recon_loss,
                     val_recon_loss)
        print(to_print)
        if epoch % 10 == 0:
            saved_model = {
                        'train_epochs': epoch + 1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'recon_criterion': criterion.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        }
            torch.save(saved_model, 'gdrive/MyDrive/pre_model{}'.format(epoch))
    return train_recon_losses, val_recon_losses

# only validate the autoencoder part of the model
def pre_validate(model, val_loader, recon_criterion, device):
    model.eval()
    total = 0
    running_recon_loss = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            features, idx_list = model.encoder.forward(images)
            recon_images = model.decoder.forward(features, idx_list)
            recon_loss = recon_criterion(recon_images, images)
            running_recon_loss += float(recon_loss.item() * labels.shape[0])
            total += images.shape[0]
            del images
            del labels
            del recon_images
            del features
            del idx_list
            del recon_loss
            torch.cuda.empty_cache()
    return running_recon_loss/total

# only train the model's classifier
def train_class(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    t_start = time.time()
    train_class_losses = []
    train_accs = []
    val_class_losses = []
    val_accs = []
    for epoch in range(epochs):
        model.train()
        running_class_loss = 0
        total = 0
        correct = 0
        for idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            features, idx_list = model.encoder.forward(images)
            features = features.detach()
            output = model.classifier.forward(features)
            class_loss = criterion(output, labels)
            class_loss.backward()
            optimizer.step()
            preds = torch.argmax(output, dim=1)
            correct += (preds == labels).sum().item()
            running_class_loss += float(class_loss.item() * images.shape[0])
            total += images.shape[0]
            del images
            del labels
            del features
            del idx_list
            del output
            del class_loss
            del preds
            torch.cuda.empty_cache()
        val_class_loss, val_acc = validate_class(model, val_loader, criterion, device)
        train_class_loss = running_class_loss/total
        train_acc = correct/total
        train_class_losses.append(train_class_loss)
        train_accs.append(train_acc)
        val_class_losses.append(val_class_loss)
        val_accs.append(val_acc)
        scheduler.step(val_class_loss)
        to_print = "Epoch: {}/{}, Training Time:{:.2f}, Trained Samples: {}/{}, Train Class Loss: {:.5f} Train Accuracy: {:.5f}, Val Class Loss: {:.5f}, Val Accuracy: {:.5f}".format(
                epoch+1, epochs, time.time()-t_start, total, len(train_loader.dataset), train_class_loss, train_acc,
                     val_class_loss, val_acc)
        print(to_print)
        if (epoch+1) % 10 == 0:
            saved_model = {
                        'train_epochs': epoch + 1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'criterion': criterion.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'train_class_losses': train_class_losses,
                        'train_accuracy': train_accs,
                        'valid_class_losses': val_class_losses,
                        'valid_accuracy': val_accs,
                        }
            torch.save(saved_model, 'gdrive/MyDrive/complete_model{}'.format(epoch+1))
    return train_class_losses, train_accs, val_class_losses, val_accs

# only validate the model's classifier
def validate_class(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_class_loss = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            features, idx_list = model.encoder.forward(images)
            features = features.detach()
            output = model.classifier.forward(features)
            class_loss = class_criterion(output, labels)
            running_class_loss += float(class_loss * labels.shape[0])
            total += labels.shape[0]
            preds = torch.argmax(output, dim=1)
            correct += (preds == labels).sum().item()
            """
            print(correct)
            print(preds)
            print(labels)
            """
            del images
            del labels
            del features
            del idx_list
            del output
            del class_loss
            del preds
            torch.cuda.empty_cache()
    return running_class_loss/total, correct/total

"""# Training and Validating Function"""

# train the multitasking model
def train(model, train_loader, val_loader, recon_criterion, class_criterion, optimizer, scheduler, epochs, device):
    t_start = time.time()
    train_recon_losses = []
    train_class_losses = []
    train_accs = []
    val_class_losses = []
    val_recon_losses = []
    val_accs = []
    for epoch in range(epochs):
        model.train()
        running_recon_loss = 0
        running_class_loss = 0
        total = 0
        correct = 0
        for idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            features, idx_list = model.encoder.forward(images)
            output = model.classifier.forward(features)
            recon_images = model.decoder.forward(features, idx_list)
            class_loss = class_criterion(output, labels)
            recon_loss = recon_criterion(recon_images, images)
            total_loss = class_loss + 10*recon_loss
            total_loss.backward()
            #class_loss.backward()
            optimizer.step()
            preds = torch.argmax(output, dim=1)
            correct += (preds == labels).sum().item()
            running_recon_loss += float(recon_loss.item() * images.shape[0])
            running_class_loss += float(class_loss.item() * images.shape[0])
            total += images.shape[0]
            #recon_loss = 0
            del images
            del labels
            del recon_images
            del features
            del idx_list
            del output
            del recon_loss
            del class_loss
            del preds
            del total_loss
            torch.cuda.empty_cache()
        val_class_loss, val_recon_loss, val_acc = validate(model, val_loader, recon_criterion, class_criterion, device)
        val_total_loss = val_recon_loss + val_class_loss
        train_recon_loss = running_recon_loss/total
        train_class_loss = running_class_loss/total
        train_total_loss = train_recon_loss + train_class_loss
        train_acc = correct/total
        train_recon_losses.append(train_recon_loss)
        train_class_losses.append(train_class_loss)
        train_accs.append(train_acc)
        val_class_losses.append(val_class_loss)
        val_recon_losses.append(val_recon_loss)
        val_accs.append(val_acc)
        scheduler.step(val_class_loss)
        to_print = "Epoch: {}/{}, Training Time:{:.2f}, Trained Samples: {}/{}, Train Total Loss: {:.5f}, Train Recon Loss: {:.5f}, Train Class Loss: {:.5f} Train Accuracy: {:.5f}, Val Total Loss: {:.5f}, Val Recon Loss: {:.5f}, Val Class Loss: {:.5f}, Val Accuracy: {:.5f}".format(
                epoch+1, epochs, time.time()-t_start, total, len(train_loader.dataset), train_total_loss, train_recon_loss, train_class_loss, train_acc,
                    val_total_loss, val_recon_loss, val_class_loss, val_acc)
        print(to_print)
        if (epoch+1) % 10 == 0:
            saved_model = {
                        'train_epochs': epoch + 1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'recon_criterion': recon_criterion.state_dict(),
                        'class_criterion': class_criterion.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'train_class_losses': train_class_losses,
                        'train_recon_losses': train_recon_losses,
                        'train_accuracy': train_accs,
                        'valid_class_losses': val_class_losses,
                        'valid_accuracy': val_accs,
                        'val_recon_losses': val_recon_losses,
                        }
            torch.save(saved_model, 'gdrive/MyDrive/complete_model{}'.format(epoch+1))
    return train_recon_losses, train_class_losses, train_accs, val_class_losses, val_recon_losses, val_accs

# validate the multitasking model
def validate(model, val_loader, recon_criterion, class_criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_recon_loss = 0
    running_class_loss = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            features, idx_list = model.encoder.forward(images)
            output = model.classifier.forward(features)
            recon_images = model.decoder.forward(features, idx_list)
            class_loss = class_criterion(output, labels)
            recon_loss = recon_criterion(recon_images, images)
            running_recon_loss += float(recon_loss.item() * labels.shape[0])
            running_class_loss += float(class_loss * labels.shape[0])
            total += labels.shape[0]
            preds = torch.argmax(output, dim=1)
            correct += (preds == labels).sum().item()
            """
            print(correct)
            print(preds)
            print(labels)
            """
            del images
            del labels
            del recon_images
            del features
            del idx_list
            del output
            del recon_loss
            del class_loss
            del preds
            torch.cuda.empty_cache()
    return running_class_loss/total, running_recon_loss/total, correct/total

# only train the autoencoder part of the model
def pretrain(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    t_start = time.time()
    train_recon_losses = []
    val_recon_losses = []
    for epoch in range(epochs):
        model.train()
        running_recon_loss = 0
        total = 0
        for idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            features, idx_list = model.encoder.forward(images)
            recon_images = model.decoder.forward(features, idx_list)
            recon_loss = criterion(recon_images, images)
            recon_loss.backward()
            optimizer.step()
            running_recon_loss += float(recon_loss.item() * images.shape[0])
            total += images.shape[0]
            del images
            del labels
            del recon_images
            del features
            del idx_list
            del recon_loss
            torch.cuda.empty_cache()
        val_recon_loss = pre_validate(model, val_loader, criterion, device)
        train_recon_loss = running_recon_loss/total
        train_recon_losses.append(train_recon_loss)
        val_recon_losses.append(val_recon_loss)
        scheduler.step(val_recon_loss)
        to_print = "Epoch: {}/{}, Training Time:{:.2f}, Trained Samples: {}/{}, Train Recon Loss: {:.5f}, Val Recon Loss: {:.5f}".format(
                epoch+1, epochs, time.time()-t_start, total, len(train_loader.dataset), train_recon_loss,
                     val_recon_loss)
        print(to_print)
        if epoch % 10 == 0:
            saved_model = {
                        'train_epochs': epoch + 1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'recon_criterion': criterion.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        }
            torch.save(saved_model, 'gdrive/MyDrive/pre_model{}'.format(epoch))
    return train_recon_losses, val_recon_losses

# only validate the autoencoder part of the model
def pre_validate(model, val_loader, recon_criterion, device):
    model.eval()
    total = 0
    running_recon_loss = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            features, idx_list = model.encoder.forward(images)
            recon_images = model.decoder.forward(features, idx_list)
            recon_loss = recon_criterion(recon_images, images)
            running_recon_loss += float(recon_loss.item() * labels.shape[0])
            total += images.shape[0]
            del images
            del labels
            del recon_images
            del features
            del idx_list
            del recon_loss
            torch.cuda.empty_cache()
    return running_recon_loss/total

# only train the model's classifier
def train_class(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    t_start = time.time()
    train_class_losses = []
    train_accs = []
    val_class_losses = []
    val_accs = []
    for epoch in range(epochs):
        model.train()
        running_class_loss = 0
        total = 0
        correct = 0
        for idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            features, idx_list = model.encoder.forward(images)
            features = features.detach()
            output = model.classifier.forward(features)
            class_loss = criterion(output, labels)
            class_loss.backward()
            optimizer.step()
            preds = torch.argmax(output, dim=1)
            correct += (preds == labels).sum().item()
            running_class_loss += float(class_loss.item() * images.shape[0])
            total += images.shape[0]
            del images
            del labels
            del features
            del idx_list
            del output
            del class_loss
            del preds
            torch.cuda.empty_cache()
        val_class_loss, val_acc = validate_class(model, val_loader, criterion, device)
        train_class_loss = running_class_loss/total
        train_acc = correct/total
        train_class_losses.append(train_class_loss)
        train_accs.append(train_acc)
        val_class_losses.append(val_class_loss)
        val_accs.append(val_acc)
        scheduler.step(val_class_loss)
        to_print = "Epoch: {}/{}, Training Time:{:.2f}, Trained Samples: {}/{}, Train Class Loss: {:.5f} Train Accuracy: {:.5f}, Val Class Loss: {:.5f}, Val Accuracy: {:.5f}".format(
                epoch+1, epochs, time.time()-t_start, total, len(train_loader.dataset), train_class_loss, train_acc,
                     val_class_loss, val_acc)
        print(to_print)
        if (epoch+1) % 10 == 0:
            saved_model = {
                        'train_epochs': epoch + 1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'criterion': criterion.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'train_class_losses': train_class_losses,
                        'train_accuracy': train_accs,
                        'valid_class_losses': val_class_losses,
                        'valid_accuracy': val_accs,
                        }
            torch.save(saved_model, 'gdrive/MyDrive/complete_model{}'.format(epoch+1))
    return train_class_losses, train_accs, val_class_losses, val_accs

# only validate the model's classifier
def validate_class(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_class_loss = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            features, idx_list = model.encoder.forward(images)
            features = features.detach()
            output = model.classifier.forward(features)
            class_loss = class_criterion(output, labels)
            running_class_loss += float(class_loss * labels.shape[0])
            total += labels.shape[0]
            preds = torch.argmax(output, dim=1)
            correct += (preds == labels).sum().item()
            """
            print(correct)
            print(preds)
            print(labels)
            """
            del images
            del labels
            del features
            del idx_list
            del output
            del class_loss
            del preds
            torch.cuda.empty_cache()
    return running_class_loss/total, correct/total

if __name__ == '__main__':
    splitfolders.ratio('test', output='val_test', seed=1337, ratio=(0, 0.5, 0.5), group_prefix=None)

    # check for GPU
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(device)

    train_loader, dev_loader, test_loader = preprocessing()

    """# Dimensionality Reduction """

    model = AutoClassifier()
    model.to(device)

    criterion = nn.MSELoss()
    epochs = 80
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=2, verbose=True)

    # train the encoder and decoder
    pre_train_loss, pre_val_loss = pretrain(model, train_loader, dev_loader, criterion, optimizer, scheduler, epochs, device)

    make_plots(pre_train_loss, pre_val_loss, "Autoencoder", "Loss")

    #compare the reconstructed image with the original image
    fixed_x = train_dataset[random.randint(1,100)][0].unsqueeze(0).to(device)
    compare_x = compare(fixed_x)

    save_image(compare_x.data.cpu(), 'sample_image.png')
    display(Image('sample_image.png', width=700, unconfined=True))

    class_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=2, verbose=True)
    epoch = 80

    train_class_losses, train_accs, val_class_losses, val_accs = train_class(model, train_loader, dev_loader, class_criterion, optimizer, scheduler, epochs, device)

    make_plots(train_accs, val_accs, "Classifier", "Accuracy")

    """# Train and Test the Multitasking Model"""

    model = AutoClassifier()
    model.to(device)

    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=2, verbose=True)
    epochs = 100

    train_recon_losses, train_class_losses, train_accs, val_class_losses, val_recon_losses, val_accs = train(model, train_loader, dev_loader, recon_criterion, class_criterion, optimizer, scheduler, epochs, device)

    _, _, test_acc = validate(model, test_loader, recon_criterion, class_criterion, device)
