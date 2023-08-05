import torch
from scipy.stats import pearsonr
from torch import nn
from torch.nn import init
import numpy as np
import random

from losses.cross_snr_loss import CrossSNRLoss
from losses.negative_pearson_loss import NegativePearsonLoss
from tools.ppg_tools import psd_compute_hr, base_bvp_hr, ibi_compute_hr


def weights_init_normal(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Sequential):
        return
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d, nn.ConvTranspose1d)):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.win_len = 256

    def frozen(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def defrost(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def init_random(self):
        self.apply(weights_init_normal)

    def num_params(self):
        return sum([_.numel() for _ in self.parameters()])

    def norm_STMap(self, data, smooth_val=1e-6):
        b, c, h, w = data.shape
        data = data.reshape([b, c, h * w])
        data_min = data.min(axis=2).values.reshape([b, c, 1])
        data_max = data.max(axis=2).values.reshape([b, c, 1])
        data = (data - data_min) / (data_max - data_min + smooth_val)
        data = data.reshape([b, c, h, w])
        return data

    def norm_BVP(self, data, smooth_val=1e-6):
        b, c, w = data.shape
        data_min = data.min(axis=2).values.reshape([b, c, 1])
        data_max = data.max(axis=2).values.reshape([b, c, 1])
        data = (2 * data - (data_min + data_max)) / (data_max - data_min + smooth_val)
        return data

    def cal_bvp_loss(self, bvp_gt, bvp_pred, hr=None, device=None,
                     loss_psd_func=None,
                     loss_neg_func=None,
                     ):
        assert bvp_gt.shape == bvp_pred.shape
        # loss_bvp
        if loss_neg_func is None:
            loss_neg_func = NegativePearsonLoss()
        win_len = bvp_gt.shape[-1]
        if len(bvp_pred.shape) == 3 or len(bvp_pred.shape) == 4:
            batch_size = bvp_pred.shape[0]
            bvp_pred = bvp_pred.view([batch_size, win_len])

        if len(bvp_gt.shape) == 3 or len(bvp_gt.shape) == 4:
            batch_size = bvp_gt.shape[0]
            bvp_gt = bvp_gt.view([batch_size, win_len])

        loss_bvp_p = loss_neg_func(bvp_gt, bvp_pred, )

        loss_bvp = loss_bvp_p

        if hr is not None:
            if loss_psd_func is None:
                loss_psd_func = CrossSNRLoss(
                    clip_length=win_len,
                    device=device,
                )
            loss_bvp_psd = loss_psd_func(
                bvp=bvp_pred,
                hr=hr,
            )
            loss_bvp += 0.2 * loss_bvp_psd[0]

        return loss_bvp

    def cal_hr_loss(self, hr_gt, hr_pred, loss_l1_func=None):
        if len(hr_gt) in [2, 3, 4]:
            hr_gt = hr_gt.view(hr_gt.shape[0])
        if len(hr_pred) in [2, 3, 4]:
            hr_pred = hr_pred.view(hr_pred.shape[0])
        if loss_l1_func is None:
            loss_l1_func = nn.L1Loss()
        loss = loss_l1_func(hr_gt, hr_pred)
        return loss

    def cal_hr_list(self, bvp, ):
        if len(bvp.shape) == 3 or len(bvp.shape) == 4:
            bvp = bvp.reshape([bvp.shape[0], self.win_len])
        bvp_hr = []
        for pb_idx in range(bvp.shape[0]):
            bvp_idx = bvp[pb_idx]
            bvp_pred_hr_idx = psd_compute_hr(BVP=bvp_idx, FS=30.0)
            # bvp_pred_hr_idx = base_bvp_hr(bvp=bvp_idx, fps=30.0)
            # bvp_pred_hr_idx = ibi_compute_hr(bvp=bvp[pb_idx], fps=30.0)

            # base_bvp_hr
            bvp_hr.append(bvp_pred_hr_idx)
        bvp_hr = np.array(bvp_hr, dtype=np.float32)
        return bvp_hr

    def cal_bpm(self, hr, win_len, fps):
        bpm = hr / 60 * win_len / fps
        return bpm

    def cal_hr_metric(self, hr_gt, hr_es, ):
        hr_gt = np.array(hr_gt, dtype=np.float32)
        hr_es = np.array(hr_es, dtype=np.float32)
        hr_err = hr_gt - hr_es

        STD = np.std(hr_err)
        MAE = np.mean(np.abs(hr_err))
        RMSE = np.sqrt(np.mean(np.square(hr_err)))
        R, _ = pearsonr(hr_es, hr_gt)

        ME = np.mean(hr_err)
        MER = np.mean(np.abs(hr_err) / hr_gt)

        return {
            'STD': float(STD),
            'MAE': float(MAE),
            'RMSE': float(RMSE),
            'R': float(R),
            'ME': float(ME),
            'MER': float(MER),
        }

    def cal_stm_cycle_loss(self, stm, hr, device, loss_psd_func=None,
                           ):
        b, c, w = stm.shape
        if loss_psd_func is None:
            loss_psd_func = CrossSNRLoss(
                clip_length=self.win_len,
                device=device,
            )

        r_ch = stm[:, 0, :].view([b, w])
        g_ch = stm[:, 1, :].view([b, w])
        b_ch = stm[:, 2, :].view([b, w])

        loss_bvp_psd = (loss_psd_func(
            bvp=r_ch,
            hr=hr.view(-1),
        )[0] + loss_psd_func(
            bvp=g_ch,
            hr=hr.view(-1),
        )[0] + loss_psd_func(
            bvp=b_ch,
            hr=hr.view(-1),
        )[0]) / 3

        return loss_bvp_psd


def demo():
    bm = BaseModel()
    # stm_gt = torch.randn(2, 3, 32, 256)
    # stm_pred = torch.randn(2, 3, 32, 256)
    # print('cal_stm_loss', bm.cal_stm_loss(stm_gt, stm_pred))

    device = torch.device('cuda:0')
    stm = torch.sin(torch.rand([32, 3, 256], device=device))
    hr = torch.tensor([60.0, 70.0], device=device).repeat([16]).view([-1])
    print('cal_stm_cycle_loss', bm.cal_stm_cycle_loss(stm, hr, device))
    # cal_stm_cycle_loss tensor(5.6584, device='cuda:0', grad_fn=<DivBackward0>)
    # cal_stm_cycle_loss tensor(5.6499, device='cuda:0', grad_fn=<AddBackward0>)
    # cal_stm_cycle_loss tensor(5.6486, device='cuda:0', grad_fn=<DivBackward0>)


if __name__ == '__main__':
    demo()

