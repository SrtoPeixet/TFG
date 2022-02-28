import torch
from torch import nn

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distances, labels):
      loss = 0 
      total = 0
      #Paralelizar
      distances = torch.from_numpy(distances).detach()
      labels = torch.from_numpy(labels).detach()
      # torch
      D_2 = torch.multiply(distances,distances)
      A = torch.multiply(D_2,labels)
      D_2_m  = (self.margin - D_2).clip(min=0)
      B = torch.multiply((1 - labels),D_2_m)

      loss = torch.sum(A+B)/2
      total = len(A) * (len(A) - 1) / 2
      
      return loss / total