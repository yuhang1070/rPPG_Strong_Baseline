import os
import joblib

from datasets import vipl
from preprocess_dt.ps_utils import spline_interpolate


def get_yuv(r_channel, g_channel, b_channel):
    # Conversion Formula
    y_channel = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
    u_channel = 128 - 0.168736 * r_channel - 0.331264 * g_channel + 0.5 * b_channel
    v_channel = 128 + 0.5 * r_channel - 0.418688 * g_channel - 0.081312 * b_channel

    return y_channel, u_channel, v_channel


def rgb2yuv(image_rgb):
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


def pre_load_data(
        path_list,
        win_len=256,
        video_path_prefix='video_1x1',
        step=10,
        use_yuv=True,
        db_name='',
        training=False,
        use_01=False,
        use_11=True,
):
    assert not (use_01 and use_11)
    trace_path_list = path_list
    trace_list = []
    video_path_list = []
    video_list = []

    if training:
        assert video_path_prefix == 'video_1x1'
        video_2x2_list = []
        # video_3x3_list = []
        # video_4x4_list = []
        # video_5x5_list = []

    if db_name in ['vipl', ]:
        hr_list = []

    nf_list = []
    total_num = 0
    index_list = []
    for pid, path in enumerate(path_list):
        trace = joblib.load(path)
        trace_list.append(trace)

        if 'cleanTrace' in path:
            video_path = path.replace('cleanTrace', video_path_prefix)
        elif 'gtTrace' in path:
            video_path = path.replace('gtTrace', video_path_prefix)
        else:
            raise Exception('Unknown path = `{}`'.format(path))

        video = joblib.load(video_path)
        if use_yuv:
            video = rgb2yuv(video)
        if use_01:
            video /= 255
        if use_11:
            video /= 255
            video = 2 * video - 1

        video_path_list.append(video_path)
        video_list.append(video)

        if training:
            video_xx = joblib.load(video_path.replace('video_1x1', 'video_2x2'))
            if use_yuv:
                video_xx = rgb2yuv(video_xx)
            if use_01:
                video_xx /= 255
            if use_11:
                video_xx /= 255
                video_xx = 2 * video_xx - 1
            video_2x2_list.append(video_xx)

            # video_xx = joblib.load(video_path.replace('video_1x1', 'video_3x3'))
            # if use_yuv:
            #     video_xx = rgb2yuv(video_xx)
            # video_3x3_list.append(video_xx)

            # video_xx = joblib.load(video_path.replace('video_1x1', 'video_4x4'))
            # if use_yuv:
            #     video_xx = rgb2yuv(video_xx)
            # video_4x4_list.append(video_xx)

            # video_xx = joblib.load(video_path.replace('video_1x1', 'video_5x5'))
            # if use_yuv:
            #     video_xx = rgb2yuv(video_xx)
            # video_5x5_list.append(video_xx)

        nf = len(trace)
        nf_list.append(nf)

        if db_name in ['vipl']:
            v_idx = int(video_path[video_path.rindex('vipl/') + 5:video_path.rindex('/')])
            hr_path = vipl.Vipl.hr_path_list[v_idx]
            gt_hr = vipl.read_gt_HR(hr_path)
            gt_hr = gt_hr.reshape([-1])

            gt_hr = spline_interpolate(gt_hr, nf)

            hr_list.append(gt_hr)

            # len_hr = len(gt_hr)
            # rate = len_hr / nf

        start_idx = 0
        while start_idx + win_len < nf:
            tmp_dic = {
                'pid': pid,
                'start_idx': start_idx,
            }

            if db_name in ['vipl']:
                hr = gt_hr[start_idx:(start_idx + win_len)]
                tmp_dic['hr'] = float(hr.mean())

            index_list.append(
                tmp_dic
            )

            total_num += 1
            start_idx += step

    res = {
        'trace_path_list': trace_path_list,
        'trace_list': trace_list,
        'video_path_list': video_path_list,
        'video_list': video_list,
        'nf_list': nf_list,
        'total_num': total_num,
        'index_list': index_list,
    }
    if training:
        res['video_2x2_list'] = video_2x2_list
        # res['video_3x3_list'] = video_3x3_list
        # res['video_4x4_list'] = video_4x4_list
        # res['video_5x5_list'] = video_5x5_list
    if db_name in ['vipl']:
        res['hr_list'] = hr_list

    return res


def demo():
    from config.parameters import OUTPUT_DIR
    p_list = [
        os.path.join(OUTPUT_DIR, 'preprocess_data/vipl/134/cleanTrace_ori.pkl'),
        os.path.join(OUTPUT_DIR, 'preprocess_data/vipl/135/cleanTrace_ori.pkl')
    ]
    res = pre_load_data(p_list, db_name='vipl', training=True, use_yuv=True, use_01=True, use_11=False)

    print(res.keys())
    print(res['video_list'][0].shape)
    import matplotlib.pyplot as plt
    import numpy as np
    print(res['video_list'][0].min())
    print(res['video_list'][0].max())
    plt.imshow((res['video_list'][0][:256] * 255).astype('uint8'))
    plt.show()
    res['video_2x2_list'][0][:, 31:32] = 0
    res['video_2x2_list'][0][:, 63:64] = 0
    res['video_2x2_list'][0][:, 95:96] = 0

    plt.imshow((res['video_2x2_list'][0][:256] * 255).astype('uint8'))
    plt.show()
    plt.imshow((res['video_2x2_list'][0][:256, :32] * 255).astype('uint8'))
    plt.show()
    plt.imshow((res['video_2x2_list'][0][:256, 32:64] * 255).astype('uint8'))
    plt.show()
    plt.imshow((res['video_2x2_list'][0][:256, 64:96] * 255).astype('uint8'))
    plt.show()
    # use_yuv=False
    # dict_keys(['trace_path_list',
    # 'trace_list', 'video_path_list',
    # 'video_list', 'nf_list', 'total_num',
    # 'index_list', 'video_2x2_list', 'hr_list'])
    # (944, 32, 3)
    # 33.564816
    # 202.13644
    return res


if __name__ == '__main__':
    demo()
