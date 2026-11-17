import torch
from torch import nn
import torch.nn.functional as F


class MarginEnLoss(nn.Module):
    def __init__(self, alpha, r):
        super(MarginEnLoss, self).__init__()
        self.alpha = alpha
        self.r = r

    def forward(self, anchor_group, positive_groups, negative_groups, Anchor_Group_label, margin_bounds):
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

            term_pos = torch.clamp(d_p ** 2 - margin_i ** 2 + self.r ** 2, min=0)
            term_neg = torch.clamp(margin_i ** 2 - d_n ** 2 + self.r ** 2, min=0)

            loss_i = self.alpha * term_pos.mean() + (1 - self.alpha) * term_neg.mean()
            total_loss += loss_i

        return total_loss / len(Anchor_Group_label)  # 
