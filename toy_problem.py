import os
import numpy as np
import matplotlib.pyplot as plt
# Torch imports
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split
from torchvision import transforms
import torchvision.models as models
# Sklearn imports 
from sklearn.metrics import pairwise_distances
# Import from local
from ContrastiveLoss import ContrastiveLoss
from DeepFashionDataset import DeepFashionDataset
from utils import display_image, get_label_matrix

root_PATH = os.getcwd()
img_dir = root_PATH + '/img/'




# We have to use the internal transformations of the pretrained Resnet18
tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224), #Random crop
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load the dataset with images in disk storage
dataset = DeepFashionDataset(annotations_file='toy_dataframe.csv',
                             img_dir=root_PATH,
                             transform=tfms
                             )

# Random split manual seed with 70 20 10 (%) length
split_size = [
            int(0.7*len(dataset.img_labels)),
            int(0.2*len(dataset.img_labels)),
            int(len(dataset.img_labels)-(int(0.7*len(dataset.img_labels)) + int(0.2*len(dataset.img_labels))))
            ]
train_dataset,val_dataset, test_dataset = random_split(dataset,split_size, generator=torch.Generator().manual_seed(23))

# TRAIN LOADER
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64, 
                                               shuffle=True)
                                        
img,label = train_dataset.__getitem__(np.random.randint(0,len(train_dataset)))

display_image(img)
fig = plt.figure()

criterion = ContrastiveLoss()
resnet18 = models.resnet18(pretrained=True)

resnet18.fc = nn.Identity() # Set last layer as Identity

if torch.cuda.is_available():
    resnet18 = resnet18.cuda()
    criterion = criterion.cuda()
    print("GPU ready to fight")

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
        torch.save(CNN.state_dict(), root_PATH + model_name)
          
    return losses_list


## TRAIN 

learning_rate = 0.01 # baixar
optimizer = torch.optim.SGD(resnet18.parameters(),lr = learning_rate, 
                            weight_decay=1e-5, momentum=0.9)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = resnet18.to(device)

losses = train(model, train_loader, optimizer, criterion, num_epochs=10, model_name='toy_model.ckpt', device=device)

# MAYBE WE COULD CLUSTER THE DATA AND USE THOSE CLASSES.
