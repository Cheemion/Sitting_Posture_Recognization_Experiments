#! /usr/bin/env python3
# PyTorch
import torchvision.models as models
import sys
import os
import torch 
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import torch 
import torchvision
# for data preprocess
import numpy as np 
import pandas as pd
from torchvision.io import read_image
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torchvision.datasets import ImageFolder
import torchvision.models as models
# For plotting  
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import figure

# parameter setting
dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
train_path = dirname + '/train'
test_path = dirname + '/test'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
myseed = 2022118  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
os.makedirs('./models', exist_ok=True)
save_path = './models/CNN_model.pth'

# hyper parameter   
batch_size = 4
n_epochs = 500
lr = 0.001
momentum = 0.9
num_workers = 4
input_image_width = 224
output_dim = 4


def train(dataloader, model, loss_fn, optimizer):
    ticks_before = time.time()
    size = len(dataloader)
    num_batches = len(dataloader)
    model.train()
    running_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        loss, current = loss.item(), batch
        running_loss = running_loss + loss
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    running_loss = running_loss / num_batches
    print(f"average one epoch loss: {running_loss:>7f}")
    ticks_after = time.time()
    return ticks_before - ticks_after

def test(dataloader, model, loss_fn):
    ticks_before = time.time()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss = test_loss + loss_fn(pred, y).item()
            correct = correct +  (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss = test_loss / num_batches
    correct = correct / size
    print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    ticks_after = time.time()
    return ticks_before - ticks_after

if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize([input_image_width, input_image_width]),transforms.ToTensor(), normalize])
    training_dataset = ImageFolder(train_path, transform=transform)
    test_dataset = ImageFolder(test_path, transform=transform)
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    #model = models.wide_resnet50_2(pretrained=True)
    model = models.googlenet(pretrained=True)
    model.fc = nn.Linear(1024, output_dim)
    model.to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.99))
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()
    consuming_time_train = 0
    consuming_time_test = 0
    for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        consuming_time_train += train(train_dataloader, model, loss_fn, optimizer)
        consuming_time_test += test(test_dataloader, model, loss_fn)
    print(f"Epoch {n_epochs}, train_time {consuming_time_train}, test_time {consuming_time_test}")
    torch.save(model.state_dict(), save_path)
