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
eval_mode = False
cluster_mode = True
'''
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
'''


if(train_mode):

    criterion = TripletLoss()
    resnet50 = models.resnet50(pretrained=True)
    resnet50.fc = nn.Identity() # Set last layer as Identity
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64, 
                                               shuffle=True)

    gpu_ready_to_fight(resnet50,criterion)
    learning_rate = 0.1 # baixar
    optimizer = torch.optim.SGD(resnet50.parameters(),lr = learning_rate, 
                                weight_decay=1e-5, momentum=0.9)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = resnet50.to(device)
    losses = train(model, train_loader, optimizer, criterion, num_epochs=15 , model_name='triplets_resnet50.pt', device=device)
    with open('triplets_loss_resnet50.npy', 'wb') as f:
        np.save(f, np.array(losses))

if(eval_mode):
    model_name =  'triplets_resnet18.pt'
    output_name = "triplets_resnet18_test_output_001.txt"

    # Set Device to CUDA and load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(models_PATH + model_name)   
    criterion = TripletLoss()
    gpu_ready_to_fight(model,criterion)
    model.eval()
    '''
    torch.cuda.empty_cache()

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

    generate_outputs(model,test_loader,device,outputs_PATH + output_name)
    
    print("Output generated correctly...")
    outputs = np.loadtxt(outputs_PATH + output_name)
    distances = pairwise_distances(X = outputs[:5001], metric = 'l2', n_jobs = -1)
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
    acc = 0 
    
    '''

    

    
  






    '''
    for i in range(5001):
       
  
       positions = get_k_closer_images_to_positions(distances,i,k)
       #print("POSITONS : ", positions)

       #fig, ax = plt.subplots(nrows=2, ncols=6)
       img_1,GT_label = test_dataset.__getitem__(i)

       #ax[0][0].imshow(torch.transpose(img_1.T,0,1))
       #ax[0][0].set_xlabel(label) 
       cnt=0
       loop = 0
       for pos in positions:
           img_1,label = test_dataset.__getitem__(pos)
           if cnt == 0:
               predicted_label=label
           else:
               predicted_label += label
           cnt +=1
           if(cnt == 6):
            predicted_label = predicted_label.astype(float) / k
            predicted_label = [truncate_label(label) for label in predicted_label]
            acc += cosine_similarity(GT_label.astype(int).reshape(1,-1),np.array(predicted_label).reshape(1,-1))
            loop+=1
            cnt = 0
            #print("True label: ",GT_label," Predicted label: ",predicted_label)
           #ax[0 + loop][cnt].imshow(torch.transpose(img_1.T,0,1))
           #plt.savefig(pairs_PATH+"CLOSER_TO_" + str(i) +".png
       if (i % 100 == 0):
           print('Predicted {}/{}'.format(i,len(test_dataset)))
       if (i % 1000 == 0):
           print('Acc {}%'.format(acc/(i+1)))
    acc = acc / len(test_dataset)
    print("Acc: " + str(acc))
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

            distances = torch.cdist(outputs,outputs,p=2).requires_grad_()

            #distance = distance_matrix(outputs.cpu().detach().numpy(),outputs.cpu().detach().numpy())
            # Forward pass
            triplets = generate_all_triplets(size=len(outputs))
            similarities = compute_similarities(labels.float())
            permutations = random.sample(triplets,len(outputs))
            loss = criterion(torch.multiply(distances,distances),similarities, permutations)

            loss_avg += loss.cpu().item()
            nBatches+=1
            losses_list.append(loss_avg / nBatches)

            if (i+1) % 1 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(1, 1, i+1, total_step, loss_avg / nBatches))

    
    with open('triplets_resnet50_test_loss.npy', 'wb') as f:
        np.save(f, np.array(losses_list))
    

if (cluster_mode):
    import pandas as pd
    model_name =  'triplets_resnet18.pt'
    output_name = "triplets_resnet18_test_output_001.txt"

    # Set Device to CUDA and load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(models_PATH + model_name)   
    criterion = TripletLoss()
    gpu_ready_to_fight(model,criterion)
    model.eval()
    
    torch.cuda.empty_cache()
    cols = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
       'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
       'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
       'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
       'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
       'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
       'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
       'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
       'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
       'Wearing_Necktie', 'Young']
    dataset = DeepFashionDataset(annotations_file=data_PATH + 'list_attr_celeba_multi.csv',
                             img_dir=img_dir)
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
    for c in range(20):
        cluster_PATH = root_PATH + '/clusters/cluster' + str(c) + '/'
        cluster_df = pd.read_csv(cluster_PATH + 'cluster'+ str(c) + '.csv')
        hist_labels = np.zeros(40)
        cnt = 0
        for value in (cluster_df.values):
            label = test_dataset.__getitem__(int(value))[1]
            hist_labels += label
            if (cnt < 30):
                img,label = test_dataset.__getitem__(int(value))
                plt.imshow(torch.transpose(img.T,0,1))
                plt.savefig(cluster_PATH + 'image_' + str(value) + '.png')
                cnt+=1

        print(hist_labels)
        norm_hist_labels = hist_labels / len(cluster_df)
        plt.figure(figsize=(12,8))
        plt.title("Cluster " + str(c + 1) + " Attributes Histogram")
        plt.bar(cols,norm_hist_labels)
        plt.xticks(rotation='vertical')
        plt.savefig(cluster_PATH + 'histogram.png')
        print(len(cluster_df))













    '''
    
    cluster = [  1,   7,  20,  28,  33,  35,  68,  80,  87,  93, 101, 107, 113,
            153, 223, 231, 248, 255, 260, 261, 278, 280, 296, 303, 320, 321,
            323, 329, 342, 350, 356, 378, 381, 388, 397, 429, 447, 448, 475,
            491, 495, 508, 519, 521, 526, 530, 531, 534, 537, 540, 542, 547,
            551, 557, 560, 574, 581, 585, 600, 605, 612, 615, 652, 654, 661,
            671, 681, 698, 707, 709, 720, 733, 734, 735, 744, 750, 776, 778,
            782, 789, 791, 803, 806, 808, 854, 862, 865, 869, 890, 891, 894,
            897, 904, 913, 930, 935, 943, 946, 957, 971]
    for c in cluster:
        img,label = test_dataset.__getitem__(c)
        plt.imshow(torch.transpose(img.T,0,1))
        plt.savefig(cluster_PATH + str(c) + '.png')

        '''