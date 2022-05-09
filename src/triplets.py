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
from TripletLoss import TripletLoss
from DeepFashionDataset import DeepFashionDataset
from utils import compute_similarities, display_image, get_label_matrix, get_pairs_of_closer, get_k_closer_images_to_positions, generate_all_triplets
import random

def generate_outputs(model, test_loader,device,output_PATH):
    with open(output_PATH,'w') as f:
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            output = model(images)
            np.savetxt(f, output.cpu().detach().numpy())
            f.write("\n")
            if (i+1) % 50 == 0:
                print('Predicted Batch [{}/{}]'.format(i,len(test_loader)))
        f.close()


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

            distances = torch.cdist(outputs,outputs,p=2).requires_grad_()

            #distance = distance_matrix(outputs.cpu().detach().numpy(),outputs.cpu().detach().numpy())
            # Forward pass
            triplets = generate_all_triplets(size=len(outputs))
            similarities = compute_similarities(labels.float())
            permutations = random.sample(triplets,len(outputs))
            loss = criterion(torch.multiply(distances,distances),similarities, permutations)
 
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
        torch.save(CNN,models_PATH + model_name)

          
    return losses_list

def gpu_ready_to_fight(resnet18,criterion):
    if torch.cuda.is_available():
        resnet18 = resnet18.cuda()
        criterion = criterion.cuda()
        print("GPU ready to fight")
# Def PATHS
root_PATH = os.getcwd()
img_dir = root_PATH + '/img_cel/img_align_celeba/'
data_PATH = root_PATH + '/data/'
models_PATH = root_PATH + '/models/'
outputs_PATH = root_PATH + '/outputs/' 
pairs_PATH = root_PATH + '/closerTriplets/'

# Set MODE
train_mode = False
eval_mode = True

    # We have to use the internal transformations of the pretrained Resnet18
tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(128),
    transforms.RandomCrop(112), #Random crop
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load the dataset with images in disk storage
dataset = DeepFashionDataset(annotations_file=data_PATH + 'list_attr_celeba.csv',
                            img_dir=img_dir,
                            transform=tfms
                            )

# Random split manual seed with 70 20 10 (%) length
split_size = [
            int(0.7*len(dataset.img_labels)),
            int(0.2*len(dataset.img_labels)),
            int(len(dataset.img_labels)-(int(0.7*len(dataset.img_labels)) + int(0.2*len(dataset.img_labels))))
            ]
train_dataset,val_dataset, test_dataset = random_split(dataset,split_size, generator=torch.Generator().manual_seed(23))



if(train_mode):

    criterion = TripletLoss()
    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc = nn.Identity() # Set last layer as Identity
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64, 
                                               shuffle=True)

    gpu_ready_to_fight(resnet18,criterion)
    learning_rate = 0.1 # baixar
    optimizer = torch.optim.SGD(resnet18.parameters(),lr = learning_rate, 
                                weight_decay=1e-5, momentum=0.9)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = resnet18.to(device)
    losses = train(model, train_loader, optimizer, criterion, num_epochs=15 , model_name='triplets.pt', device=device)
    with open('triplets_loss.npy', 'wb') as f:
        np.save(f, np.array(losses))

if(eval_mode):

    model_name =  'triplets.pt'
    output_name = "triplets_output_001.npy"

    # Set Device to CUDA and load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(models_PATH + model_name)   
    criterion = TripletLoss()
    gpu_ready_to_fight(model,criterion)
    model.eval()

    tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # LOAD DATASET WITHOUT TRANSFORMS
    dataset = DeepFashionDataset(annotations_file=data_PATH + 'list_attr_celeba.csv',
                             img_dir=img_dir, transform=tfms)
    # Random split manual seed with 70 20 10 (%) length
    split_size = [
                int(0.7*len(dataset.img_labels)),
                int(0.2*len(dataset.img_labels)),
                int(len(dataset.img_labels)-(int(0.7*len(dataset.img_labels)) + int(0.2*len(dataset.img_labels))))
                ]
    train_dataset,val_dataset, test_dataset = random_split(dataset,split_size, generator=torch.Generator().manual_seed(23))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=64, 
                                               shuffle=False)
    
    #generate_outputs(model,test_loader,device,outputs_PATH + output_name)
    print("Output generated correctly...")
    outputs = np.loadtxt(outputs_PATH + output_name)
    distances = pairwise_distances(X = outputs, metric = 'l2', n_jobs = -1)

    score = 0
    k=10
    # LOAD DATASET WITHOUT TRANSFORMS 
    dataset = DeepFashionDataset(annotations_file=data_PATH + 'list_attr_celeba.csv',
                             img_dir=img_dir)
    # Random split manual seed with 70 20 10 (%) length
    split_size = [
                int(0.7*len(dataset.img_labels)),
                int(0.2*len(dataset.img_labels)),
                int(len(dataset.img_labels)-(int(0.7*len(dataset.img_labels)) + int(0.2*len(dataset.img_labels))))
                ]
    train_dataset,val_dataset, test_dataset = random_split(dataset,split_size, generator=torch.Generator().manual_seed(23))
 
    for i in range(len(test_dataset)):
       
       image,true_label = test_dataset.__getitem__(i)
  
       positions = get_k_closer_images_to_positions(distances,i,k)
       print("POSITONS : ", positions)
       predicted_label = 0

       fig, ax = plt.subplots(nrows=2, ncols=6)
       img_1,label = test_dataset.__getitem__(i)

       ax[0][0].imshow(torch.transpose(img_1.T,0,1))
       ax[0][0].set_xlabel(label) 
       cnt=0
       loop = 0
       for pos in positions:
           cnt +=1
           img_1,label = test_dataset.__getitem__(pos)
           if(cnt == 6):
            loop+=1
            cnt = 0
           ax[0 + loop][cnt].imshow(torch.transpose(img_1.T,0,1))
           plt.savefig(pairs_PATH+"CLOSER_TO_" + str(i) +".png")
    
       print("True label: ",true_label," Predicted label: ",predicted_label)
       if (i % 100 == 0):
           print('Predicted {}/{}'.format(i,len(test_dataset)))
    accuracy = score / len(test_dataset)
    print("Acc: " + str(accuracy))
    print(zeros,ones)