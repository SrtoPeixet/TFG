import os
from posixpath import normpath
import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt

# Torch imports
import torch

from torch import nn
from torch.nn import functional as F
from torch.nn.modules.distance import PairwiseDistance

from torch.utils.data import Dataset,random_split

from torchvision import transforms
from torchvision.io import read_image

import torchvision.models as models
from torch.optim import Adam, SGD


from pytorch_metric_learning import losses
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances

from pathlib import Path

root_PATH = os.getcwd()

img_dir = os.path.abspath('./')
SPRINT_path = str(root_PATH) + ''



def display_image(img):
  plt.imshow(torch.transpose(img.T,0,1))

class DeepFashionDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        print(image)
        label = self.img_labels.iloc[idx, 1:].to_numpy(dtype="int8")
        if self.transform:
            image = self.transform(image)
        return image, label

# We have to use the internal transformations of the pretrained Resnet18

tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224), #Random crop
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

dataset = DeepFashionDataset(annotations_file='toy_dataframe.csv',
                             img_dir=img_dir,
                             transform=tfms
                             )



# Random split manual seed with 70 20 10 (%) length
split_size = [int(0.7*len(dataset.img_labels)),int(0.2*len(dataset.img_labels)),int(0.1*len(dataset.img_labels))+1]

train_dataset,val_dataset, test_dataset = random_split(dataset,split_size, generator=torch.Generator().manual_seed(23))

# TRAIN LOADER

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64, 
                                               shuffle=True)
                                        
img,label = train_dataset.__getitem__(np.random.randint(0,len(train_dataset)))
display_image(img)
print(label)
fig = plt.figure()


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distances, labels):
      m = 5
      loss = 0 
      total = 0
      #Paralelizar
      distances = torch.from_numpy(distances).detach()
      labels = torch.from_numpy(labels).detach()
      # torch
      D_2 = torch.multiply(distances,distances)
      A = torch.multiply(D_2,labels)
      D_2_m  = (5 - D_2).clip(min=0)
      B = torch.multiply(D_2_m,(1 - labels))

      loss = torch.sum(A+B)/2
      total = len(A) * (len(A) - 1) / 2
      
      return loss / total
      

criterion = ContrastiveLoss()

resnet18 = models.resnet18(pretrained=True)

resnet18.fc = nn.Identity()
if torch.cuda.is_available():
    resnet18 = resnet18.cuda()
    criterion = criterion.cuda()
    print("GPU ready to fight")


def get_label_matrix(labels):
  labels = labels.cpu().detach().numpy()
  lab_mat = np.empty([len(labels),len(labels)])
  for i,lab in enumerate(labels):
    for j,lab_2 in enumerate(labels):
      if lab == lab_2:
        lab_mat[i][j] = 0
      else:
        lab_mat[i][j] = 1
  return lab_mat

def train(CNN, train_loader, optimizer,criterion, num_epochs, model_name='model.ckpt', device='cpu'):
    CNN.train() # Set the model in train mode
    total_step = len(train_loader)
    losses_list = []
    criterion = criterion
    # Iterate over epochs
    for epoch in range(num_epochs):
        # Iterate the dataset
        loss_avg = 0
        nBatches = 0
        for i, (images, labels) in enumerate(train_loader):
            # Get batch of samples and labels
            images = images.to(device)
            labels = labels.type(torch.LongTensor).to(device)
            outputs = CNN(images)

            distance = 1-pairwise_distances(outputs.cpu().detach().numpy(), metric="cosine")

            #distance = distance_matrix(outputs.cpu().detach().numpy(),outputs.cpu().detach().numpy())
            # Forward pass
            loss = criterion(distance,get_label_matrix(labels)).requires_grad_()
 
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg += loss.cpu().item()
            nBatches+=1
            
            if (i+1) % 50 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss_avg / nBatches))
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss_avg / nBatches))
        losses_list.append(loss_avg / nBatches)
        torch.save(CNN.state_dict(), SPRINT_path+ '/' + model_name)
          
    return losses_list


## TRAIN 

learning_rate = 0.01 # baixar
optimizer = torch.optim.SGD(resnet18.parameters(),lr = learning_rate, 
                            weight_decay=1e-5, momentum=0.9)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = resnet18.to(device)

losses = train(model, train_loader, optimizer, criterion, num_epochs=10, model_name='toy_model.ckpt', device=device)

# MAYBE WE COULD CLUSTER THE DATA AND USE THOSE CLASSES.
