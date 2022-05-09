import torch
from torch import nn

class TripletLoss(nn.Module):
    
    # loss =  Y*1/2*DW^2 + (1 - Y)*1/2*max(0,m-DW)^2
    def __init__(self):
        super(TripletLoss, self).__init__()
    

    def forward(self, distances,similarities,permutations):
        loss = 0
        for i in range(len(distances)):
            # i is our anchor image
            sample_1 = permutations[i][1]
            sample_2 = permutations[i][2]
            '''
            print("-----------", i, sample_1, sample_2)
            print("Distances: ", distances[i][sample_1], distances[i][sample_2])
            print("Similarities: ", similarities[i][sample_1], similarities[i][sample_2])
            '''
            if (similarities[i][sample_1] >= similarities[i][sample_2]):
                # sample_1 is the positive sample
                loss += (distances[i][sample_1] - distances[i][sample_2] + 100).clip(min=0)
            else: 
                
                # sample_2 is the positive sample
                loss += (distances[i][sample_2] - distances[i][sample_1] + 100).clip(min=0)

        return loss/len(distances)
    '''

    def forward(self, distances,similarities):
      
        anchor_positive_dist = torch.unsqueeze(distances, 2)
        ancohr_negative_dist = torch.unsqueeze(distances, 1)

        triplet_loss = anchor_positive_dist - anchor_positive_dist
        triplet_loss = torch.clamp(triplet_loss, min=0)
        valid_trìplets = torch.greater(triplet_loss,1e-16)
        num_valid = torch.sum(valid_trìplets)
        print(num_valid)
        print(anchor_positive_dist.shape, ancohr_negative_dist.shape, triplet_loss.shape )
        return

    def forward(self, distances,similarities):
      loss = 0
      total_steps = 0
      for i in range(len(distances)):
        for j in range(len(distances)):
            for k in range(len(distances)):
                if not ((i == j) | (i == k) | ( j==k )):
                    if similarities[i][j] >= similarities[i][k]: 
                        loss += (distances[i][j] - distances[i][k]).clip(min=0)
                    else:
                        loss +=   (distances[i][k] - distances[i][j]).clip(min=0)
                    total_steps +=1 
      return (loss/total_steps)
    '''
  