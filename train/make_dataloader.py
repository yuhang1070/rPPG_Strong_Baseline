import torch
from torch.utils.data import DataLoader, Dataset
import joblib
from torchvision import transforms
import numpy as np
import random

from solutions.dr.train.load_data import pre_load_data
from tools.ppg_tools import psd_compute_hr, base_bvp_hr, ibi_compute_hr


def aug_rnd_sl(total_len, min_mask_len, max_mask_len):
    while True:
        rnd_start = random.randint(0, total_len)
        rnd_len = random.randint(min_mask_len, max_mask_len)
        if rnd_start + rnd_len < total_len:
            return rnd_start, rnd_len


def aug_rnd_noise_frame(data, num_frames, min_mask_len=1, max_mask_len=16):
    # random noise
    rnd = random.random()
    if rnd < 0.5:
        if rnd < 0.25:
            rnd_len = random.randint(min_mask_len, max_mask_len)
            rnd_indices = np.random.randint(0, num_frames, size=(rnd_len,))
            data[rnd_indices] = np.random.random([rnd_len, data.shape[1], data.shape[2]])
        else:
            rnd_start, rnd_len = aug_rnd_sl(num_frames, min_mask_len, max_mask_len)
            data[rnd_start:rnd_start + rnd_len] = np.random.random([rnd_len, data.shape[1], data.shape[2]])
    return data


def aug_rnd_erase_frame(data, num_frames, min_mask_len=1, max_mask_len=16):
    # random erase
    rnd = random.random()
    if rnd < 0.5:
        if rnd < 0.25:
            rnd_len = random.randint(min_mask_len, max_mask_len)
            rnd_indices = np.random.randint(0, num_frames, size=(rnd_len,))
            data[rnd_indices] = 0.0
        else:
            rnd_start, rnd_len = aug_rnd_sl(num_frames, min_mask_len, max_mask_len)
            data[rnd_start:rnd_start + rnd_len] = 0.0
    return data


def aug_rnd_noise_roi(data, roi_nums, min_mask_len=1, max_mask_len=3):
    # random noise
    rnd = random.random()
    if rnd < 0.5:
        if rnd < 0.25:
            rnd_len = random.randint(min_mask_len, max_mask_len)
            rnd_indices = np.random.randint(0, roi_nums, size=(rnd_len,))
            data[:, rnd_indices, :] = np.random.random([data.shape[0], rnd_len, data.shape[2]])
        else:
            rnd_start, rnd_len = aug_rnd_sl(roi_nums, min_mask_len, max_mask_len)
            data[:, rnd_start:rnd_start + rnd_len, :] = np.random.random([data.shape[0], rnd_len, data.shape[2]])
    return data


def aug_rnd_erase_roi(data, roi_nums, min_mask_len=1, max_mask_len=3):
    # random erase
    rnd = random.random()
    if rnd < 0.5:
        if rnd < 0.25:
            rnd_len = random.randint(min_mask_len, max_mask_len)
            rnd_indices = np.random.randint(0, roi_nums, size=(rnd_len,))
            data[:, rnd_indices, :] = 0.0
        else:
            rnd_start, rnd_len = aug_rnd_sl(roi_nums, min_mask_len, max_mask_len)
            data[:, rnd_start:rnd_start + rnd_len, :] = 0.0
    return data


def aug_random_shuffle_roi(data, ):
    rnd = random.random()
    if rnd < 0.5:
        indices = list(range(data.shape[1]))
        random.shuffle(indices)
        return data[:, indices, :]
    else:
        return data


def auto_hf_roi(data, ):
    indices = list(range(0, 5))[::-1] + list(range(5, 10)[::-1]) + \
              list(range(10, 16)[::-1]) + list(range(16, 22)[::-1]) + \
              list(range(22, 28)[::-1]) + [29, 28, 31, 30]
    return data[:, indices, :]


def aug_hf_roi(data, ):
    rnd = random.random()
    if rnd < 0.5:
        return auto_hf_roi(data)
    return data


class PpgDataset(Dataset):
    def __init__(
            self,
            path_list: list,
            training: bool = False,
            transform=None,
            num_frames=256,
            step=10,
            db_name='vipl',
            video_path_prefix='video_1x1',
            which_aug='',
            which_rm='',
    ):
        self.path_num = len(path_list)
        self.path_list = path_list
        self.training = training
        self.transform = transform
        self.db_name = db_name
        self.num_frames = num_frames
        self.step = step
        self.which_aug = which_aug
        self.video_path_prefix = video_path_prefix
        self.which_rm = which_rm

        self.use_yuv = True
        self.use_01 = False
        self.use_11 = True
        assert not (self.use_01 and self.use_11)

        self.res = pre_load_data(
            path_list=self.path_list,
            win_len=self.num_frames,
            video_path_prefix=video_path_prefix,
            step=step,
            use_yuv=self.use_yuv,
            db_name=db_name,
            training=training,
            use_01=self.use_01,
            use_11=self.use_11,
        )
        self.trace_path_list = self.res['trace_path_list']
        self.trace_list = self.res['trace_list']
        self.video_path_list = self.res['video_path_list']
        self.video_list = self.res['video_list']
        self.nf_list = self.res['nf_list']
        self.total_num = self.res['total_num']
        self.index_list = self.res['index_list']

        if self.db_name in ['vipl']:
            self.hr_list = self.res['hr_list']

        if self.training:
            self.video_2x2_list = self.res['video_2x2_list']
            # self.video_3x3_list = self.res['video_3x3_list']
            # self.video_4x4_list = self.res['video_4x4_list']
            # self.video_5x5_list = self.res['video_5x5_list']

    def __len__(self):
        return self.total_num

    def __getitem__(self, index):
        item = self.index_list[index]
        pid = item['pid']
        start_idx = item['start_idx']
        if (
                self.which_rm in ['rm'] and
                self.training and
                self.nf_list[pid] - start_idx >= (256 + 20) and start_idx >= 20
        ):
            start_idx += random.randint(-10, 11)
        #
        t_video_list = self.video_list
        if self.which_aug not in [None, '']:
            rnd = random.random()
            if rnd < 0.5:
                t_video_list = self.video_list
            else:
                t_video_list = self.video_2x2_list
            # elif rnd < 0.5 + 0.5 / 3 * 2:
            #     t_video_list = self.video_3x3_list
            # else:
            #     t_video_list = self.video_4x4_list
            # t_video_list = self.video_3x3_list
            # t_video_list = self.video_4x4_list
            # t_video_list = self.video_5x5_list

            # elif rnd < 0.625:
            #     t_video_list = self.video_2x2_list
            # elif rnd < 0.75:
            #     t_video_list = self.video_3x3_list
            # elif rnd < 0.875:
            #     t_video_list = self.video_4x4_list
            # else:
            #     t_video_list = self.video_5x5_list
        video = t_video_list[pid]
        trace = self.trace_list[pid]
        part_video = video[start_idx:start_idx + self.num_frames]
        part_trace = trace[start_idx:start_idx + self.num_frames]

        use_2x2 = part_video.shape[1] > 127
        # print(part_video.shape)
        # (256, 32, 3)
        if self.which_aug not in [None, ''] and use_2x2:
            # random crop
            assert self.which_aug in ['msrca']
            max_num = part_video.shape[1] - 33
            rnd_start = random.randint(0, max_num)
            part_video = part_video[:, rnd_start: rnd_start + 32, :]
            # if self.which_aug in ['msra']:
            #     rnd = random.random()
            #     if rnd < 0.75:
            #         # random crop
            #         max_num = part_video.shape[1] - 33
            #         rnd_start = random.randint(0, max_num)
            #         part_video = part_video[:, rnd_start: rnd_start + 32, :]
            #     else:
            #         # random select
            #         beta = part_video.shape[1] // 32
            #         rnd_se = np.random.randint(0, beta, 32) + np.arange(0, 32) * beta
            #         part_video = part_video[:, rnd_se, :]
            # elif self.which_aug in ['msrsa']:
            #     # random select
            #     beta = part_video.shape[1] // 32
            #     rnd_se = np.random.randint(0, beta, 32) + np.arange(0, 32) * beta
            #     part_video = part_video[:, rnd_se, :]
            # else:
            #     # random crop
            #     assert self.which_aug in ['msrca']
            #     max_num = part_video.shape[1] - 33
            #     rnd_start = random.randint(0, max_num)
            #     part_video = part_video[:, rnd_start: rnd_start + 32, :]

        part_video = part_video.transpose([2, 1, 0, ])

        # print('part_video.min', part_video.min())
        # print('part_video.max', part_video.max())
        # exit()
        # if not use_2x2 and self.training:
        #     part_video = aug_hf_roi(part_video)

        if self.db_name in ['vipl']:
            if self.training and self.which_rm in ['rm']:
                gt_hr = self.hr_list[pid]
                gt_hr_part = gt_hr[start_idx: start_idx + self.num_frames]
                hr = float(gt_hr_part.mean())
            else:
                hr = item['hr']
        else:
            # hr = ibi_compute_hr(bvp=part_trace, fps=30.0)
            # hr = base_bvp_hr(bvp=part_trace, fps=30.0)
            hr = psd_compute_hr(BVP=part_trace, FS=30.0)

        # num_frames height channel
        # channel height num_frames
        # if not self.training:
        #     part_video = auto_hf_roi(part_video)

        return torch.tensor(part_video, dtype=torch.float32), \
               torch.tensor(part_trace, dtype=torch.float32), \
               torch.tensor(hr, dtype=torch.float32)


def make_train_loader(
        train_path_list: list,
        batch_size: int = 2,
        step=10,
        video_path_prefix='video_1x1',
        which_aug='',
        db_name='',
        which_rm='',
):
    train_dataset = PpgDataset(
        path_list=train_path_list,
        training=True,
        transform=None,
        step=step,
        video_path_prefix=video_path_prefix,
        which_aug=which_aug,
        db_name=db_name,
        which_rm=which_rm,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    return train_loader


def make_valid_loader(
        valid_path_list: list,
        batch_size: int = 2,
        step=10,
        video_path_prefix='video_1x1',
        db_name='',
        shuffle=False,
):
    valid_dataset = PpgDataset(
        path_list=valid_path_list,
        training=False,
        transform=None,
        step=step,
        video_path_prefix=video_path_prefix,
        db_name=db_name,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=4,
    )
    return valid_loader


if __name__ == '__main__':
    pass
