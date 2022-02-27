#! /usr/bin/env python3
# PyTorch
import sys
import os
import torch 
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
# for data preprocess
import numpy as np 
import pandas as pd
import csv
import time
# For plotting  
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import figure

# parameter setting
train_path = '/angels_train.csv'
test_path = '/angels_test.csv'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
myseed = 2022118  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
os.makedirs('./models', exist_ok=True)
save_path = './models/BP_model.pth'

# hyper parameter
batch_size = 100
n_epochs = 200
lr = 0.001
momentum = 0.9
num_workers = 4
class BPDataset(Dataset):
    def __init__(self, path):
        #Read data into numpy
        dirname, filename = os.path.split(os.path.abspath(sys.argv[0])) 
        with open(dirname + path, 'r') as fp:
            data = list(csv.reader(fp)) #读取所有的数据
            data = np.array(data[1:])[:, 1:].astype(float)
        feature_num = list(range(data.shape[1] - 1))
        lables = data[:, -1]  # 结果 比如说covid病人数目
        data = data[:, feature_num] #去掉了target这一行的数据, 这个就是训练集了
        self.data = torch.FloatTensor(data)  #最后的需要训练的数据
        self.target = torch.LongTensor(lables) #最后的对比的数据
        # Normalize features (you may remove this part to see what will happen)
        #self.data[:, 40:] = (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) / self.data[:, 40:].std(dim=0, keepdim=True)
        self.input_dim = self.data.shape[1]
        self.output_dim = len(set(lables))
        print('BP Dataset ({} samples found, each input_dim = {}, output_dim = {})'.format( len(self.data), self.input_dim, self.output_dim))

    def __getitem__(self, index):
        return self.data[index], self.target[index]
    def __len__(self):
        return len(self.data)

def prep_dataloader(path, batch_size, n_jobs = 0):
    dataset = BPDataset(path)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=False, num_workers=n_jobs)
    return dataloader, dataset.input_dim, dataset.output_dim

class BPNet(nn.Module):
    def __init__(self, input_dim, output_dim, level1, level2):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, level1)
        self.fc2 = nn.Linear(level1,level2)
        self.fc3 = nn.Linear(level2,output_dim)
        self.drop = nn.Dropout(0.5)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        return x

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

#opt_SGD=torch.optim.SGD(net_SGD.parameters(),lr=LR)
#opt_Momentum=torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
#opt_RMSprop=torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9)
#opt_Adam=torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
if __name__ == '__main__':
    train_dataloader, input_dim, output_dim = prep_dataloader(train_path, batch_size, num_workers)
    test_dataloader, _, _ = prep_dataloader(test_path, batch_size, num_workers)
    model = BPNet(input_dim, output_dim, 49, 29)
    model = model.to(device)
    print(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.99))
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
