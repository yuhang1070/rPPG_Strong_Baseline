import os
import joblib
from tools.ppg_tools import psd_compute_hr, ibi_compute_hr
from scipy import interpolate
import numpy as np


def spline_interpolate(sig, target_len):
    _x = np.arange(len(sig))
    interp_func = interpolate.splrep(x=_x, y=sig, k=3)
    ix3 = np.linspace(_x[0], _x[-1], target_len)
    sig_s = interpolate.splev(x=ix3, tck=interp_func)
    return sig_s


# np.seterr(divide='ignore', invalid='ignore')


def normalize_video_data(data, smooth_val=1e-6):
    # data = data.copy()
    # data[np.isnan(data)] = 0.
    # data[np.isinf(data)] = 0.
    win_len = data.shape[0]
    roi_num = data.shape[1]
    data = data.reshape([win_len, roi_num * 3])  # WIN_LEN ROI_NUM 3 (256, 44, 3)
    data = data.transpose([1, 0])  # ROI_NUM * 3 WIN_LEN (132, 256)

    # print(data.min(axis=1))
    # print(data.max(axis=1))

    data_min = data.min(axis=1).reshape([roi_num * 3, 1])
    data_max = data.max(axis=1).reshape([roi_num * 3, 1])
    data = (data - data_min) / (data_max - data_min + smooth_val)

    data = data.transpose([1, 0])
    data = data.reshape([win_len, roi_num, 3])

    return data


def get_yuv(r_channel, g_channel, b_channel):
    # Conversion Formula
    y_channel = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
    u_channel = 128 - 0.168736 * r_channel - 0.331264 * g_channel + 0.5 * b_channel
    v_channel = 128 + 0.5 * r_channel - 0.418688 * g_channel - 0.081312 * b_channel

    return y_channel, u_channel, v_channel


def my_rgb2yuv(image_rgb):
    b_channel = image_rgb[:, :, 2]
    g_channel = image_rgb[:, :, 1]
    r_channel = image_rgb[:, :, 0]
    y_channel, u_channel, v_channel = get_yuv(
        r_channel=r_channel,
        g_channel=g_channel,
        b_channel=b_channel,
    )
    image_rgb[:, :, 0] = y_channel
    image_rgb[:, :, 1] = u_channel
    image_rgb[:, :, 2] = v_channel

    return image_rgb


def load_video(video_dir: str, num_frames: int, which_skin_threshold=0):
    pic_names = ['{}_{}.pkl'.format(_, which_skin_threshold) for _ in range(num_frames)]

    video_1x1 = []
    video_2x2 = []
    video_3x3 = []
    video_4x4 = []
    video_5x5 = []
    for pic_name in pic_names:
        pic_path = os.path.join(video_dir, pic_name)
        if os.path.isfile(pic_path):
            pic = joblib.load(pic_path)  # bgr (5, 5, 3) float32 0 <= val <= 255
            #
            # print(pic.keys())
            pic_1x1 = pic['1x1'].reshape([32 * 1 * 1, 3])
            pic_2x2 = pic['2x2'].reshape([32 * 2 * 2, 3])
            pic_3x3 = pic['3x3'].reshape([32 * 3 * 3, 3])
            pic_4x4 = pic['4x4'].reshape([32 * 4 * 4, 3])
            pic_5x5 = pic['5x5'].reshape([32 * 5 * 5, 3])
        else:
            print('Not exist: `{}`'.format(pic_path))
            pic_1x1 = np.zeros(shape=[32 * 1 * 1, 3], dtype=np.float32)
            pic_2x2 = np.zeros(shape=[32 * 2 * 2, 3], dtype=np.float32)
            pic_3x3 = np.zeros(shape=[32 * 3 * 3, 3], dtype=np.float32)
            pic_4x4 = np.zeros(shape=[32 * 4 * 4, 3], dtype=np.float32)
            pic_5x5 = np.zeros(shape=[32 * 5 * 5, 3], dtype=np.float32)
            # raise
        #
        video_1x1.append(pic_1x1)
        video_2x2.append(pic_2x2)
        video_3x3.append(pic_3x3)
        video_4x4.append(pic_4x4)
        video_5x5.append(pic_5x5)

    return video_1x1, video_2x2, video_3x3, video_4x4, video_5x5
    # return video_1x1, video_2x2


def moving_average_3(x, ):
    res = np.convolve(x, np.ones(3), 'same') / 3
    res[0] = np.mean(x[:2])
    res[-1] = np.mean(x[-2:])
    return res


def save_data(
        save_dir: str,
        video_1x1,
        video_2x2,
        # video_3x3,
        # video_4x4,
        # video_5x5,
        bvp: np.ndarray,
        fps: float,
        clip_length: int = 300,
        step_size: int = 10,
        which_skin_threshold=0,
):
    assert len(bvp) == len(video_1x1)
    assert len(bvp) == len(video_2x2)
    # assert len(bvp) == len(video_3x3)
    # assert len(bvp) == len(video_4x4)
    # assert len(bvp) == len(video_5x5)

    # num_frames = len(video)
    # lenHr = len(hr)
    # rate = lenHr / num_frames

    start_idx = 0
    start_cnt = 0

    # plt.imshow(video[:64].astype('int32'))
    # plt.show()
    video_1x1 = my_rgb2yuv(video_1x1)
    video_2x2 = my_rgb2yuv(video_2x2)
    # video_3x3 = my_rgb2yuv(video_3x3)
    # video_4x4 = my_rgb2yuv(video_4x4)
    # video_5x5 = my_rgb2yuv(video_5x5)

    # plt.imshow(video[:64].astype('int32'))
    # plt.show()
    # exit(0)

    video_1x1 /= 255
    video_2x2 /= 255
    # video_3x3 /= 255
    # video_4x4 /= 255
    # video_5x5 /= 255

    # print(video.shape)
    # plt.imshow(video[:300])
    # plt.show()
    # exit(0)

    # print(video.min())
    # print(video.max())
    # video[video < 0.] = 0.
    # video[video > 1.] = 1.
    # print(video.min())
    # print(video.max())
    # assert 0 <= video.min() <= 1
    # assert 0 <= video.max() <= 1

    while True:
        ori_data_1x1 = video_1x1[start_idx:start_idx + clip_length].copy()
        video_chip_1x1 = video_1x1[start_idx:start_idx + clip_length].copy()
        ori_data_2x2 = video_2x2[start_idx:start_idx + clip_length].copy()
        video_chip_2x2 = video_2x2[start_idx:start_idx + clip_length].copy()
        # ori_data_3x3 = video_3x3[start_idx:start_idx + clip_length].copy()
        # video_chip_3x3 = video_3x3[start_idx:start_idx + clip_length].copy()
        # ori_data_4x4 = video_4x4[start_idx:start_idx + clip_length].copy()
        # video_chip_4x4 = video_4x4[start_idx:start_idx + clip_length].copy()
        # ori_data_5x5 = video_5x5[start_idx:start_idx + clip_length].copy()
        # video_chip_5x5 = video_5x5[start_idx:start_idx + clip_length].copy()
        if len(video_chip_1x1) != clip_length:
            break

        video_chip_1x1 = normalize_video_data(video_chip_1x1)
        video_chip_2x2 = normalize_video_data(video_chip_2x2)
        # video_chip_3x3 = normalize_video_data(video_chip_3x3)
        # video_chip_4x4 = normalize_video_data(video_chip_4x4)
        # video_chip_5x5 = normalize_video_data(video_chip_5x5)

        bvp_chip = bvp[start_idx:start_idx + clip_length].copy()

        bvp_chip_min = bvp_chip.min()
        bvp_chip_max = bvp_chip.max()
        bvp_chip = (bvp_chip - bvp_chip_min) / (bvp_chip_max - bvp_chip_min)

        ibi_hr = ibi_compute_hr(bvp=bvp_chip, fps=fps, )
        psd_hr = psd_compute_hr(BVP=bvp_chip, FS=fps, LL_PR=40.0, UL_PR=180.0)

        print('ibi_hr: {}, psd_hr: {}'.format(ibi_hr, psd_hr, ))
        # if abs(psd_hr - ibi_hr) >= 50:
        #     plt.plot(bvp_chip)
        #     plt.show()
        #     exit(0)
        # continue

        # exit(0)
        # plt.xlabel('ibi_hr: {}, psd_hr: {}'.format(ibi_hr, psd_hr))
        # plt.plot(bvp_chip)
        # plt.show()

        save_cnt_path = os.path.join(
            save_dir,
            # '{}.pkl'.format(str(uuid.uuid4())),
            '{}_{}.pkl'.format(start_cnt, which_skin_threshold),
        )

        # print('psd_hr: {}, ibi_hr = {}'.format(psd_hr, ibi_hr))
        print('save_cnt_path: {}'.format(save_cnt_path))
        save_dict = {
            'fps': fps,
            'ibi_hr': ibi_hr,
            'psd_hr': psd_hr,
            'bvp': bvp_chip,
            'norm_data_1x1': video_chip_1x1,
            'norm_data_2x2': video_chip_2x2,
            # 'norm_data_3x3': video_chip_3x3,
            # 'norm_data_4x4': video_chip_4x4,
            # 'norm_data_5x5': video_chip_5x5,

            'ori_data_1x1': ori_data_1x1,
            'ori_data_2x2': ori_data_2x2,
            # 'ori_data_3x3': ori_data_3x3,
            # 'ori_data_4x4': ori_data_4x4,
            # 'ori_data_5x5': ori_data_5x5,
        }
        joblib.dump(save_dict, save_cnt_path)

        start_idx += step_size
        start_cnt += 1
