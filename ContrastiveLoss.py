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
      #Paralelizar
      distances = torch.from_numpy(distances).detach()
      labels = torch.from_numpy(labels).detach()
      # torch
      D_2 = torch.multiply(distances,distances).requires_grad_()
      A = 0.5 * torch.multiply(D_2,labels).requires_grad_()

      D_margin_clipped  = (self.margin - distances).clip(min=0).requires_grad_()
      D_2_MC = torch.multiply(D_margin_clipped,D_margin_clipped).requires_grad_()
      B = 0.5 * torch.multiply((1 - labels),D_2_MC).requires_grad_()

      loss = torch.sum(A+B).requires_grad_()
      total = len(A) * (len(A) - 1) / 2
      
      return loss / total