import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

"""
Reference:
    https://github.com/nxsEdson/CVD-Physiological-Measurement/blob/master/utils/loss/loss_SNR.py
    https://github.com/radimspetlik/hr-cnn/blob/master/cmp/nrppg/torch/TorchLossComputer.py
    https://github.com/terbed/Deep-rPPG/blob/master/src/errfuncs.py
"""


class CrossSNRLoss(nn.Module):
    def __init__(self, clip_length=256, delta=3, loss_type=7, device=None, batch_size=32, fps=30.0):
        """
        bpm: beat per minute
        T: clip_length
        HR = BPM / (T / fps) *  60
        """
        super(CrossSNRLoss, self).__init__()
        self.low_bound = 40
        self.high_bound = 150
        self.pi = 3.14159265
        self.clip_length = clip_length
        self.delta = delta
        self.loss_type = loss_type
        self.device = device
        self.batch_size = batch_size
        self.fps = fps
        self.cross_entropy = nn.CrossEntropyLoss()

        # bpm_range
        bpm_range = torch.arange(
            self.low_bound, self.high_bound, dtype=torch.float32, device=self.device
        ).div_(60.0).repeat(batch_size, 1)
        self.bpm_range = Variable(bpm_range, requires_grad=False)
        # f_t
        self.f_t = (self.bpm_range / self.fps).view(self.batch_size, -1, 1)
        # two_pi_n
        self.two_pi_n = Variable(
            2 * self.pi * torch.arange(0, self.clip_length, dtype=torch.float32, device=self.device),
            requires_grad=False,
        ).repeat(batch_size, 1).view(batch_size, 1, -1)
        # ftt
        self.sin_ftt = torch.sin(self.f_t * self.two_pi_n)
        self.cos_ftt = torch.cos(self.f_t * self.two_pi_n)
        # hanning
        self.hanning = Variable(
            torch.from_numpy(np.hanning(self.clip_length)).type(torch.FloatTensor),
            requires_grad=True,
        ).view(1, -1).to(self.device)

    def forward(self, bvp, hr, ):
        # hr = bpm * (fps * 60 / self.clip_length)
        hr = 1 * hr
        hr[hr.ge(self.high_bound)] = self.high_bound - 1
        hr[hr.le(self.low_bound)] = self.low_bound

        preds = (bvp * self.hanning).view(self.batch_size, 1, -1)
        complex_absolute = torch.sum(preds * self.sin_ftt, dim=-1) ** 2 + torch.sum(preds * self.cos_ftt, dim=-1) ** 2

        target = hr - self.low_bound
        target = target.type(torch.long).view(self.batch_size)

        whole_max_val, whole_max_idx = complex_absolute.max(1)
        whole_max_idx = whole_max_idx + self.low_bound

        if self.loss_type == 1:
            loss = self.cross_entropy(complex_absolute, target)

        elif self.loss_type == 7:
            norm_t = (torch.ones(self.batch_size).cuda() / torch.sum(complex_absolute, dim=1))
            norm_t = norm_t.view(-1, 1)
            complex_absolute = complex_absolute * norm_t
            loss = self.cross_entropy(complex_absolute, target)

            idx_l = target - self.delta
            idx_l[idx_l.le(0)] = 0
            idx_r = target + self.delta
            idx_r[idx_r.ge(self.high_bound - self.low_bound - 1)] = self.high_bound - self.low_bound - 1
            loss_snr = 0.0
            for i in range(0, self.batch_size):
                loss_snr = loss_snr + 1 - torch.sum(complex_absolute[i, idx_l[i]:idx_r[i]])
            loss_snr = loss_snr / self.batch_size

            loss = loss + loss_snr
        else:
            raise Exception('Unknown loss_type = `{}`'.format(self.loss_type))

        return loss, whole_max_idx
