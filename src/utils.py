import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics.pairwise import pairwise_distances


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
    print(distances)
    #distances = distance_matrix(outputs,outputs)
    distances = np.triu(distances)

    positions = get_closer_images(distances,100)
    print(positions)
    cnt = 0
    for pos in positions:      
          fig, ax = plt.subplots(nrows=1, ncols=2)
          img_1 = test_dataset.__getitem__(pos[0])[0]
          img_2 = test_dataset.__getitem__(pos[1])[0]
          print(pos[0],pos[1])
          cnt+=1
          ax[0].imshow(torch.transpose(img_1.T,0,1))
          ax[1].imshow(torch.transpose(img_2.T,0,1))
          fig.suptitle('Pairs of most closer images' + str(cnt), fontsize=16)
          plt.savefig(pairs_PATH+"Top_" + str(cnt) +"_most_closer_images.png")