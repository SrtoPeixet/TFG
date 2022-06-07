import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics.pairwise import pairwise_distances
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split
from torchvision import transforms
import torchvision.models as models


def display_image(img):
    '''
    This function displays an torch tensor as an image.
    img: Torch Tensor 
    '''
    plt.imshow(torch.transpose(img.T,0,1))

def get_label_matrix(labels):
  lab_mat = torch.empty([len(labels),len(labels)])
  for i,lab in enumerate(labels):
    for j,lab_2 in enumerate(labels):
      if lab == lab_2:
        lab_mat[i][j] = 1
      else:
        lab_mat[i][j] = 0
  return lab_mat
def compute_similarities(labels):
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    similarities = torch.empty(len(labels),len(labels))
    i = 0
    for row in labels:
        j = 0
        for row_2 in labels:
            if(i >= j):
                sim =  cos(row,row_2)
                similarities[i][j] = sim
                similarities[j][i] = sim
            j+=1
        i+=1
    return similarities

def get_triplet_permutation(size,anchor_permutation=None):
    
    permutation = []
    if anchor_permutation:

        return
    else: 
        for i in range(size):


            permutation[i] = random_val



def get_closer_images(distances,k):
  '''
  This function returns the positions of minimum distances in distance matrix.
  Set 0 to inf, find minimum and get its x,y coords. Then, set it to inf and repeat.
  '''
  distances[distances == 0] = np.inf
  positions = []
  for i in range(k):
    position = np.where(distances == np.min(distances))
    position = [position[0][0],position[1][0]]
    positions.append(position)
    distances[position[0]][position[1]] = np.inf    
  return positions

def get_pairs_of_closer(test_dataset,output_PATH,pairs_PATH):

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    outputs = np.loadtxt(output_PATH)
    distances = pairwise_distances(X = outputs, metric = 'l2', n_jobs = -1)
    distances = np.triu(distances)

    positions = get_closer_images(distances,100)
    print(positions)
    cnt = 0
    for pos in positions:      
          fig, ax = plt.subplots(nrows=1, ncols=2)
          img_1 = test_dataset.__getitem__(pos[0])[0]
          img_2 = test_dataset.__getitem__(pos[1])[0]
          cnt+=1
          ax[0].imshow(torch.transpose(img_1.T,0,1))
          ax[1].imshow(torch.transpose(img_2.T,0,1))
          fig.suptitle('Pairs of most closer images' + str(cnt), fontsize=16)
          plt.savefig(pairs_PATH+"Top_" + str(cnt) +"_most_closer_images.png")

def get_k_closer_images_to_positions(distances, position,k=10):
    '''
    outputs_PATH : location of outputs
    position: row of image we want to compare.
    '''
    
    vector = distances[position][:]
    vector[vector <1e-15] = np.inf

    positions = []
    for i in range(k):
        val, idx = min((val, idx) for (idx, val) in enumerate(vector))
        positions.append(idx)
        vector[idx] = np.inf
    return positions
    
def generate_all_triplets(size=64):
    result = []
    for i in range(size):
      for j in range(size):
        for z in range(size):
          if ( (i != j) & (i!= z) & (z!= j)):
            A = [i,j,z]
            result.append(A)
    return result

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
            if (i+1) % 50 == 0:
                print('Predicted Batch [{}/{}]'.format(i,len(test_loader)))
        f.close()

def gpu_ready_to_fight(resnet18,criterion):
    if torch.cuda.is_available():
        resnet18 = resnet18.cuda()
        criterion = criterion.cuda()
        print("GPU ready to fight")

def truncate_label(label):
    if label < 0:
        return -1
    else:
        return 1

def get_hard_triplets(similarities):
    similarities = 1 - similarities
    result = []
    for i in range(len(similarities)):
        vector = similarities[i][:]
        vector[vector <1e-15] = np.inf
        positions = [i]
        val, idx = min((val, idx) for (idx, val) in enumerate(vector))
        positions.append(idx)
        vector[idx] = np.inf
        val, idx = min((val, idx) for (idx, val) in enumerate(vector))
        positions.append(idx)
        vector[idx] = np.inf
        result.append(positions)
    return result