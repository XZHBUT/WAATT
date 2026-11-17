import torch
from torch import nn
import torch.nn.functional as F

class ImprovedTripletLoss(nn.Module):
    def __init__(self, alpha):
        super(ImprovedTripletLoss, self).__init__()
        self.alpha = alpha


    def forward(self, anchor_group, positive_groups, negative_groups, Anchor_Group_label,margin_bounds):

        total_loss = 0.0
        for index, i_class in enumerate(Anchor_Group_label):
            pos = positive_groups[index]  # (B, 256)
            neg = negative_groups[index]  # (B, 256)
            anchor = anchor_group[index].unsqueeze(0)  # (1, 256)

            # 
            d_p = F.pairwise_distance(anchor.expand_as(pos), pos, p=2)
            d_n = F.pairwise_distance(anchor.expand_as(neg), neg, p=2)

            # 
            margin_i = margin_bounds[i_class]
            term_pos = d_p / (1 + torch.exp(-(d_p - margin_i)))
            term_neg = torch.clamp(margin_i - d_n, min=0) / (1 + torch.exp(-(margin_i - d_n)))

            loss_i = self.alpha * term_pos.mean() + (1 - self.alpha) * term_neg.mean()
            total_loss += loss_i

        return total_loss / len(Anchor_Group_label)  # 


class ImprovedTripletLoss_2(nn.Module):
    def __init__(self, alpha):
        super(ImprovedTripletLoss_2, self).__init__()
        self.alpha = alpha

    def forward(self, anchor_group, positive_groups, negative_groups, Anchor_Group_label, margin_bounds):

        sample_total_loss = 0.0
        for index, i_class in enumerate(Anchor_Group_label):
            pos = positive_groups[index]  # (B, 256)
            neg = negative_groups[index]  # (B, 256)
            anchor = anchor_group[index].unsqueeze(0)  # (1, 256)
            # 
            d_p = F.pairwise_distance(anchor.expand_as(pos), pos, p=2)
            d_n = F.pairwise_distance(anchor.expand_as(neg), neg, p=2)
            # 
            margin_i = margin_bounds[i_class]
            term_pos = d_p / (1 + torch.exp(-(d_p - margin_i)))
            term_neg = torch.clamp(margin_i - d_n, min=0) / (1 + torch.exp(-(margin_i - d_n)))
            loss_i = self.alpha * term_pos.mean() + (1 - self.alpha) * term_neg.mean()
            sample_total_loss += loss_i

        sample_total_loss = sample_total_loss / len(Anchor_Group_label)

        anchor_dist_loss = 0.0
        Anchor_total_sum = sum(range(1, len(Anchor_Group_label)))
        for i in range(len(Anchor_Group_label)):
            for j in range(i + 1, len(Anchor_Group_label)):
                anchor_i = anchor_group[i].unsqueeze(0)  #  i
                anchor_j = anchor_group[j].unsqueeze(0)  #  j

                # 
                d_ij = F.pairwise_distance(anchor_i, anchor_j, p=2)

                # 
                margin_sum = margin_bounds[Anchor_Group_label[i]] + margin_bounds[Anchor_Group_label[j]]
                # ，
                # ，
                term_neg_ij = torch.clamp(margin_sum - d_ij, min=0) / (1 + torch.exp(-(margin_sum - d_ij)))
                anchor_dist_loss += term_neg_ij.mean()
        anchor_dist_loss = anchor_dist_loss / Anchor_total_sum

        total_loss = 0.0 * (anchor_dist_loss) + sample_total_loss

        return total_loss  # 