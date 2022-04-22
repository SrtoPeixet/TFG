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
from utils import display_image, get_label_matrix, get_pairs_of_closer, get_k_closer_images_to_positions

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

            distances = torch.cdist(outputs,outputs,p=2)

            #distance = distance_matrix(outputs.cpu().detach().numpy(),outputs.cpu().detach().numpy())
            # Forward pass
            loss = criterion(distances,get_label_matrix(labels).to(device).requires_grad_())
 
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

def generate_outputs(model, test_loader,device,output_PATH):
    with open(output_PATH,'w') as f:
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            output = model(images)
            np.savetxt(f, output.cpu().detach().numpy())
            f.write("\n")
            if (i+1) % 15 == 0:
                print('Predicted Batch [{}/{}]'.format(i,len(test_loader)))
        f.close()

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
pairs_PATH = root_PATH + '/pairs_no_transforms/'

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
#Change transforms during testing

# Load the dataset with images in disk storage
dataset = DeepFashionDataset(annotations_file=data_PATH + 'blond_bald_df.csv',
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
## TRAIN 
if(train_mode):

    criterion = ContrastiveLoss()
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
    losses = train(model, train_loader, optimizer, criterion, num_epochs=30 , model_name='blond_bald.pt', device=device)
    with open('blond_bald_loses_train.npy', 'wb') as f:
        np.save(f, np.array(losses))


## EVALUATE
if(eval_mode):
    tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    #Change transforms during testing

    # Load the dataset with images in disk storage
    dataset = DeepFashionDataset(annotations_file=data_PATH + 'blond_bald_df.csv',
                                img_dir=img_dir,
                                transform=tfms
                                )

    model_name =  'blond_bald.pt'
    output_name = "outputs_blond_bald.npy"

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=64, 
                                               shuffle=False)
    # Set Device to CUDA and load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(models_PATH + model_name)   
    criterion = ContrastiveLoss()
    gpu_ready_to_fight(model,criterion)
    model.eval()

    
    loss_avg = 0
    total_step = len(test_dataset)/64
    losses_list = []
    nBatches = 0
    '''
    for i, (images, labels) in enumerate(test_loader):
            # Get batch of samples and labels
            images = images.to(device)
            labels = labels.type(torch.LongTensor).to(device)
            outputs = model(images)

            distances = torch.cdist(outputs,outputs,p=2)

            #distance = distance_matrix(outputs.cpu().detach().numpy(),outputs.cpu().detach().numpy())
            # Forward pass
            loss = criterion(distances,get_label_matrix(labels).to(device).requires_grad_())

            loss_avg += loss.cpu().item()
            nBatches+=1
            losses_list.append(loss_avg / nBatches)

            if (i+1) % 1 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(1, 1, i+1, total_step, loss_avg / nBatches))
   
    
    with open('blond_bald_loses_test.npy', 'wb') as f:
        np.save(f, np.array(losses_list))

    '''

    generate_outputs(model,test_loader,device,outputs_PATH + output_name)
    print("Output generated correctly...")

    # LOAD DATASET WITHOUT TRANSFORMS
    dataset = DeepFashionDataset(annotations_file=data_PATH + 'blond_bald_df.csv',
                             img_dir=img_dir)
    # Random split manual seed with 70 20 10 (%) length
    split_size = [
                int(0.7*len(dataset.img_labels)),
                int(0.2*len(dataset.img_labels)),
                int(len(dataset.img_labels)-(int(0.7*len(dataset.img_labels)) + int(0.2*len(dataset.img_labels))))
                ]
    train_dataset,val_dataset, test_dataset = random_split(dataset,split_size, generator=torch.Generator().manual_seed(23))
    
    outputs = np.loadtxt(outputs_PATH + output_name)
    distances = pairwise_distances(X = outputs, metric = 'l2', n_jobs = -1)

    score = 0
    k=10
 
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
           print("LABEL KNN: ",test_dataset.__getitem__(pos)[1][0])
           predicted_label+=test_dataset.__getitem__(pos)[1][0]/k
           
           cnt +=1
           img_1,label = test_dataset.__getitem__(pos)
           if(cnt == 6):
            loop+=1
            cnt = 0
           ax[0 + loop][cnt].imshow(torch.transpose(img_1.T,0,1))
           #ax[0 + loop][cnt].set_xlabel(label) 
           #plt.savefig(pairs_PATH+"Example_" + str(i) +".png")
           
       predicted_label = round(predicted_label)
       print("True label: ",true_label," Predicted label: ",predicted_label)
       if true_label == predicted_label:
           score+=1
       if (i % 100 == 0):
           print('Predicted {}/{}'.format(i,len(test_dataset)))
    accuracy = score / len(test_dataset)
    print("Acc: " + str(accuracy))
    print(zeros,ones)
    
              
    


   
    #get_pairs_of_closer(test_dataset,outputs_PATH + output_name,pairs_PATH)
    #print("Pairs of closer images generated correctly...")





#TODO : Compute accuracy after training. How ? 
#TODO : Inverse transform pairs of images
