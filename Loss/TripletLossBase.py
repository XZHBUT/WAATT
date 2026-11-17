import torch
from torch import nn


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)  # triplet

    def forward(self, inputs, labels):

        n = inputs.size(0)  # batch_size，inputs,batchsize * 
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)  # ， （keepdim2），nxn
        dist = dist + dist.t()  # dis[i][j]ij
        dist.addmm_( inputs, inputs.t(), beta=1, alpha=-2)  # 2 i*j  (a-b)^2
        dist = dist.clamp(min=1e-12).sqrt()  # 

        # For each anchor, find the hardest positive and negative
        mask = labels.expand(n, n).eq(labels.expand(n, n).t())  # mask[i][j] = Trueijlabel， =Falseijlabel
        # print(mask)
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # ilabelj
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # ilabelj
        dist_ap = torch.cat(dist_ap)  # listtensortensor
        dist_an = torch.cat(dist_an)
        # print(dist_ap)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)  # dist_anshape1tensor
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

if __name__ == '__main__':
    input_seq = torch.randn(8, 1024)  # 
    labels = torch.randint(0, 3, (8,),dtype=torch.float32)
    print(labels)

    Loss = TripletLoss()

    a = Loss(input_seq, labels)
    print(a)
    a.item()