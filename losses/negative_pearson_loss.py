import torch
import torch.nn as nn


class NegativePearsonLoss(nn.Module):
    def __init__(self):
        super(NegativePearsonLoss, self).__init__()

    def forward(self, x, y):
        if x.size() != y.size():
            raise Exception('`x` and `y` MUST have same size!!!')
        elif len(x.size()) != 2:
            raise Exception('`len(x.size())` MUST equal 2!!!')
        tot_loss = 1 - torch.mean(torch.cosine_similarity(x, y, dim=1))
        return tot_loss
