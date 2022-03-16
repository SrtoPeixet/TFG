import torch
from torch import nn

class ContrastiveLoss(nn.Module):
    
    # loss =  Y*1/2*DW^2 + (1 - Y)*1/2*max(0,m-DW)^2
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distances, labels):
      loss = 0 
      total = 0
      # torch
      D_2 = torch.multiply(distances,distances) #similar
      A = 0.5 * torch.multiply(D_2,labels)

      D_margin_clipped  = (self.margin - distances).clip(min=0) #dissimilar
      D_2_MC = torch.multiply(D_margin_clipped,D_margin_clipped)
      B = 0.5 * torch.multiply((1 - labels),D_2_MC)

      loss = torch.sum(A+B)
      total = len(A) * len(A)
      
      return loss / total