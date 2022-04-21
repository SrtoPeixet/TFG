import pandas as pd
import os
from PIL import Image
from DeepFashionDataset import DeepFashionDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


root_PATH = os.getcwd()
img_dir = root_PATH + '/img/'
data_PATH = root_PATH + '/data/'
hacienda_PATH = root_PATH + '/hacienda/'


tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.RandomCrop(224), #Random crop
    transforms.ToTensor(),
    ])

# Load the dataset with images in disk storage
dataset = DeepFashionDataset(annotations_file=data_PATH + 'leather_floral.csv',
                            img_dir=root_PATH,
                            transform=tfms
                            )

for i in range(8000,len(dataset.img_labels)):
    
    img,label = dataset.__getitem__(i)
    plt.imshow(torch.transpose(img.T,0,1))
    plt.xlabel(label)

    plt.savefig(hacienda_PATH + str(i) + ".png")

