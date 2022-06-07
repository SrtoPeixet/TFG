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
from sklearn.metrics import pairwise_distances, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
# Import from local
from TripletLoss import TripletLoss
from DeepFashionDataset import DeepFashionDataset
from utils import compute_similarities, display_image, get_label_matrix, get_pairs_of_closer, get_k_closer_images_to_positions, generate_all_triplets, truncate_label
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
            loss = criterion(outputs,labels.type(torch.float))
 
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

# Use threshold to define predicted labels and invoke sklearn's metrics with different averaging strategies.
def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }






class Multilabelresnet18(nn.Module):
    def __init__(self, n_classes=40):
        super().__init__()
        resnet18 = models.resnet18(pretrained=True)
        resnet18.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet18.fc.in_features, out_features=n_classes)
            )
        self.base_model = resnet18
        self.sigm = nn.Sigmoid()

    def forward(self, x):
       return self.sigm(self.base_model(x))








# Def PATHS
root_PATH = os.getcwd()
img_dir = root_PATH + '/img_cel/img_align_celeba/'
data_PATH = root_PATH + '/data/'
models_PATH = root_PATH + '/models/'
outputs_PATH = root_PATH + '/outputs/' 
pairs_PATH = root_PATH + '/closerMulti/'

# We have to use the internal transformations of the pretrained resnet18

train_mode = True
eval_mode = False

if(train_mode):

    tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(128),
    transforms.RandomCrop(112), #Random crop
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Load the dataset with images in disk storage
    dataset = DeepFashionDataset(annotations_file=data_PATH + 'list_attr_celeba_multi.csv',
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


    model = Multilabelresnet18(40)
    model.train()


    criterion = nn.BCELoss()

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=64, 
                                                shuffle=True)

    gpu_ready_to_fight(model,criterion)
    learning_rate = 0.1 # baixar
    optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate, 
                                weight_decay=1e-5, momentum=0.9)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    losses = train(model, val_loader, optimizer, criterion, num_epochs=15 , model_name='multilabel_resnet18.pt', device=device)
    with open('val_loss_multilabel_resnet18.npy', 'wb') as f:
        np.save(f, np.array(losses))

if(eval_mode):
    model_name =  'multilabel_resnet18.pt'
    output_name = "triplets_loss_multilabel_resnet18_outputs_001.txt"

    # Set Device to CUDA and load model
    model = Multilabelresnet18(40)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(models_PATH + model_name)   
    criterion = nn.BCELoss()
    gpu_ready_to_fight(model,criterion)
    model.eval()

    tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # LOAD DATASET WITHOUT TRANSFORMS
    dataset = DeepFashionDataset(annotations_file=data_PATH + 'list_attr_celeba_multi.csv',
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
    #generate_outputs(model,test_loader,device,outputs_PATH + output_name) SET TRANSFORMS WHEN COMMENT OUT

    print("Output generated correctly...")
    #outputs = np.loadtxt(outputs_PATH + output_name)
    '''
    similarities = pairwise_distances(X = outputs, metric = 'cosine', n_jobs = -1)

    for i in range(len(test_dataset)):
        
        fig, ax = plt.subplots(nrows=2, ncols=6)
        image,true_label = test_dataset.__getitem__(i)
        ax[0][0].imshow(torch.transpose(image.T,0,1))
        positions = get_k_closer_images_to_positions(similarities,i, 10)
        print("POSITIONS:", positions)
        cnt=0
        loop = 0
        for pos in positions:
            
            img_1,label = test_dataset.__getitem__(pos)
            cnt +=1
            if(cnt == 6):
                loop+=1
                cnt = 0
            ax[0 + loop][cnt].imshow(torch.transpose(img_1.T,0,1))
            plt.savefig(pairs_PATH+"10_CLOSER_TO_" + str(i) +".png")



  

    model.eval()
    with torch.no_grad():
        model_result = []
        targets = []
        for imgs, batch_targets in test_loader:
            imgs = imgs.to(device)
            model_batch_result = model(imgs)
            model_result.extend(model_batch_result.cpu().numpy())
            targets.extend(batch_targets.cpu().numpy())
    result = calculate_metrics(np.array(model_result), np.array(targets))
    print(result)

    '''
    
    loss_avg = 0
    total_step = len(test_dataset)/64
    losses_list = []
    nBatches = 0
    for i, (images, labels) in enumerate(test_loader):
            # Get batch of samples and labels
            images = images.to(device)
            labels = labels.type(torch.LongTensor).to(device)
            outputs = model(images)

            loss = criterion(outputs,labels.type(torch.float))


            loss_avg += loss.cpu().item()
            nBatches+=1
            losses_list.append(loss_avg / nBatches)

            if (i+1) % 1 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(1, 1, i+1, total_step, loss_avg / nBatches))

    
    with open('multilabel_resnet18_test_loss.npy', 'wb') as f:
        np.save(f, np.array(losses_list))
    
    