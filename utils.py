import torch
import matplotlib.pyplot as plt
import numpy as np

def display_image(img):
    '''
    This function displays an torch tensor as an image.
    img: Torch Tensor 
    '''
    plt.imshow(torch.transpose(img.T,0,1))

def get_label_matrix(labels):
  labels = labels.cpu().detach().numpy()
  lab_mat = np.empty([len(labels),len(labels)])
  for i,lab in enumerate(labels):
    for j,lab_2 in enumerate(labels):
      if lab == lab_2:
        lab_mat[i][j] = 1
      else:
        lab_mat[i][j] = 0
  return lab_mat