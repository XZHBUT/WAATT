import torch
from torch import nn

from tool.DataProcess import read_OrginCWRU
from tool.SampleBuild import BuildNoiseSample, BuildTriadGroup
from tool.addnoise import addNoise


class TripletLossNoise(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLossNoise, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)  # triplet

    def forward(self, Anchor_Group, Positive_Group, Negative_Group, Anchor_Group_label):
        n_class, L = Anchor_Group.shape
        # 
        AP_distances = []
        AN_distances = []
        for index, i_class in enumerate(Anchor_Group_label):
            anchor = Anchor_Group[index].unsqueeze(0)  #  i ， (1, 1024)

            #  i 
            i_P_samples = Positive_Group[index]  #  (X, 1024)
            i_N_samples = Negative_Group[index]  #  (Y, 1024)

            #  (X, 1024)
            AP_distance = torch.cdist(anchor, i_P_samples, p=2).squeeze(0)  #  (X)

            #  (Y, 1024)
            AN_distance = torch.cdist(anchor, i_N_samples, p=2).squeeze(0)  #  (Y)

            #  (X, Y) ，
            AP_distance_expanded = AP_distance.unsqueeze(1).expand(-1, AN_distance.size(0))  #  (X, Y)
            AN_distance_expanded = AN_distance.unsqueeze(0).expand(AP_distance.size(0), -1)  #  (X, Y)
            #  X * Y 
            AP_distances.append(AP_distance_expanded.flatten())  # (X * Y)
            AN_distances.append(AN_distance_expanded.flatten())  # (X * Y)

        # 
        AP_dist = torch.cat(AP_distances)  #  ()
        AN_dist = torch.cat(AN_distances)  #  ()
        # print(AP_dist.shape) # torch.Size([21600])
        # print(AN_dist.shape) # torch.Size([21600])

        # Compute ranking hinge loss
        Y = torch.ones_like(AP_dist)  # dist_anshape1tensor

        loss = self.ranking_loss(AN_dist, AP_dist, Y)
        # print(loss)

        return loss


if __name__ == '__main__':
    Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = read_OrginCWRU(filepath='../data/CWRU',
                                                                        SampleLength=1024,
                                                                        SampleNum=40,
                                                                        normal=False,
                                                                        Rate=[0.5, 0.3, 0.2]
                                                                        )
    X_noisy_samples, Y_noisy_samples = BuildNoiseSample(Test_X, Test_Y, min_snr=-4, max_snr=4, n_Noise=10,
                                                        num_class=10)

    Test_X = torch.tensor(Test_X, dtype=torch.float32)
    Test_Y = torch.tensor(Test_Y, dtype=torch.int64)

    X_noisy_samples = torch.tensor(X_noisy_samples, dtype=torch.float32)
    Y_noisy_samples = torch.tensor(Y_noisy_samples, dtype=torch.int64)

    # K(batch)
    indices = torch.randperm(Test_X.size(0))[:20]  # K

    # 
    X_random = Test_X[indices]
    Y_random = Test_Y[indices]

    Anchor_Group, Positive_Group, Negative_Group, Anchor_Group_label = BuildTriadGroup(X_random, Y_random,
                                                                                       X_noisy_samples, Y_noisy_samples,
                                                                                       num_classes=10, NK=100)
    Loss = TripletLossNoise()
    Loss(Anchor_Group, Positive_Group, Negative_Group, Anchor_Group_label)
