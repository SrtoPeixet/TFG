import torch
from torch import nn
import torchvision.models as models
import os
import numpy as np
from scipy.spatial import distance_matrix
from utils import get_closer_images
import matplotlib.pyplot as plt
import os




def get_pairs_of_closer(test_dataset,output_PATH):

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    outputs = np.loadtxt(output_PATH)
    distances = distance_matrix(outputs,outputs)
    distances = np.triu(distances)

    positions = get_closer_images(distances,25)

    cnt = 0
    for pos in positions:      
          fig, ax = plt.subplots(nrows=1, ncols=2)
          img_1 = test_dataset.__getitem__(positions[cnt][0])[0]
          img_2 = test_dataset.__getitem__(positions[cnt][1])[0]
          cnt+=1
          ax[0].imshow(torch.transpose(img_1.T,0,1))
          ax[1].imshow(torch.transpose(img_2.T,0,1))
          fig.suptitle('Pairs of most closer images' + str(cnt), fontsize=16)
          plt.savefig("Top_" + cnt +"_most_closer_images.png")
