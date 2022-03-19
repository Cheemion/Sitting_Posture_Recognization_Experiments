#! /usr/bin/env python3
# PyTorch
import sys
import os
import torch 
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
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
batch_size = 100
n_epochs = 200
lr = 0.001
momentum = 0.9
num_workers = 4
input_image_width = 132
output_dim = 4

class CNNDataset(Dataset):
    def __init__(self, annotations_file, transform = None):
        #Read data into numpy
        dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
        img_dir = dirname + '/images'
        self.img_labels = pd.read_csv(dirname + annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        print('CNN Dataset ({} samples found)'.format(len(self.img_labels)))

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.img_labels)

class CNNNetwork(nn.Module):
    def __init__(self):
        super(CNNNetwork, self).__init__()
        #First 132 * 132 * 3 
        self.conv1 = nn.Conv2d(3, 17, 3)
        self.conv2 = nn.Conv2d(17, 32, 3)
        #Second 128 * 128 * 32 
        # do maxpooling (2,2)
        #Thrid 64 * 64 * 32 
        self.conv3 = nn.Conv2d(32, 48, 3)
        self.conv4 = nn.Conv2d(48, 64, 3)
        #Fourth 60 * 60 * 64
        # maxpooling (2,2)
        #fifth 30 * 30 * 64
        self.conv5 = nn.Conv2d(64, 96, 3)
        self.conv6 = nn.Conv2d(96, 128, 3)
        #sixth 26 * 26 * 128
        #do maxpooling(2,2)
        #senventh 13 * 13 * 128
        self.conv7 = nn.Conv2d(128, 128, 3)
        self.conv8 = nn.Conv2d(128, 128, 3)
        #nineth 9 * 9 * 128
        #do maxpolling(2,2)
        #tenth 5 * 5 * 128
        self.fc1 = nn.Linear(5*5*128, 700)
        self.fc2 = nn.Linear(700, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.max_pool2d(x, (2, 2), ceil_mode = True)
        # 拉直
        x = x.view(-1, self.num_flat_features(x))        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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
    transform = transforms.Compose([transforms.Resize([input_image_width, input_image_width]), transforms.ToTensor()])
    training_dataset = ImageFolder(train_path, transform=transform)
    test_dataset = ImageFolder(test_path, transform=transform)
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    model = CNNNetwork()
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
