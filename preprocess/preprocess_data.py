from datasets import pure
from datasets import ubfc2
from datasets import vipl
from datasets import mahnob

from config.parameters import OUTPUT_DIR
from preprocess_dt.ps_utils import load_video, spline_interpolate
from tools.io_tools import mkdir_if_missing
from tools.sig_tools import butter_bandpass_filter

import joblib
import os
import cv2
import numpy as np
from tools.video_tools import num_frames_video, load_half_video_bgr

which_dataset = 'mahnob'

if which_dataset == 'ubfc2':
    video_path_list = ubfc2.UBFC2.video_path_list
elif which_dataset == 'pure':
    video_path_list = pure.Pure.video_dir_list
elif which_dataset == 'vipl':
    video_path_list = vipl.Vipl.video_path_list
elif which_dataset == 'mahnob':
    video_path_list = mahnob.Mahnob.video_path_list
else:
    raise Exception('Unknown which_dataset="{}"'.format(which_dataset))


def gen_dataset(which_dataset, which_skin_threshold, ):
    load_dir = os.path.join(
        OUTPUT_DIR,
        'preprocess_crop',
        which_dataset,
    )

    output_dir = os.path.join(
        OUTPUT_DIR,
        'preprocess_data',
        which_dataset,
    )

    for v_idx, video_path in enumerate(video_path_list):
        # if not (v_idx >= 401):
        #     continue
        print('-' * 35 + str(v_idx) + '-' * 35)
        load_video_dir = os.path.join(
            load_dir,
            str(v_idx),
        )
        output_video_dir = os.path.join(
            output_dir,
            str(v_idx),
        )
        mkdir_if_missing(output_video_dir)
        print('load_video_dir = "{}"'.format(load_video_dir))
        print('output_video_dir = "{}"'.format(output_video_dir))
        # num_frames = num_frames_video(video_path)
        if which_dataset == 'pure':
            num_frames = pure.num_frames_video(video_path)
        elif which_dataset in ['vipl', 'ubfc2']:
            num_frames = num_frames_video(video_path)
        elif which_dataset in ['mahnob']:
            num_frames = len(load_half_video_bgr(video_path))
        else:
            raise Exception('Unknown which_dataset="{}"'.format(which_dataset))

        video_1x1, video_2x2, video_3x3, video_4x4, video_5x5 = load_video(
            load_video_dir,
            num_frames=num_frames,
            which_skin_threshold=which_skin_threshold
        )
        # video_1x1, video_2x2 = load_video(
        #     load_video_dir,
        #     num_frames=num_frames,
        #     which_skin_threshold=which_skin_threshold,
        # )
        print(num_frames)
        video_1x1 = np.array(video_1x1)
        video_2x2 = np.array(video_2x2)
        video_3x3 = np.array(video_3x3)
        video_4x4 = np.array(video_4x4)
        video_5x5 = np.array(video_5x5)
        # outlier removal
        video_1x1[np.isnan(video_1x1)] = 0.
        video_1x1[np.isinf(video_1x1)] = 0.
        video_2x2[np.isnan(video_2x2)] = 0.
        video_2x2[np.isinf(video_2x2)] = 0.
        video_3x3[np.isnan(video_3x3)] = 0.
        video_3x3[np.isinf(video_3x3)] = 0.
        video_4x4[np.isnan(video_4x4)] = 0.
        video_4x4[np.isinf(video_4x4)] = 0.
        video_5x5[np.isnan(video_5x5)] = 0.
        video_5x5[np.isinf(video_5x5)] = 0.

        if which_dataset == 'pure':
            gt_path = pure.Pure.json_path_list[v_idx]
            gtTrace, _ = pure.read_signal_file(gt_path)
            video_fps = 30.0
        elif which_dataset == 'ubfc2':
            gt_path = ubfc2.UBFC2.gt_path_list[v_idx]
            gtTrace, gtHR, gtTime = ubfc2.read_signal_file(gt_path)
            video_fps = (len(gtTime) - 1) / (gtTime[-1] - gtTime[0])
        elif which_dataset == 'vipl':
            gtTrace_path = vipl.Vipl.wave_path_list[v_idx]
            gtTrace = vipl.read_wave(gtTrace_path)
            video_fps = vipl.Vipl.fps_list[v_idx]
        elif which_dataset == 'mahnob':
            gtTrace_path = mahnob.Mahnob.BDF_PATH_LIST[v_idx]
            gtTrace, pulse_fps = mahnob.read_signal_file(gtTrace_path)
            video_fps = 30.5
        else:
            raise Exception('Unknown which_dataset="{}"'.format(which_dataset))

        print('video_fps: {}'.format(video_fps))
        # print(video.shape) # (1547, 20, 3)
        video_1x1 = video_1x1.reshape([num_frames, 1 * 1 * 32, 3])
        video_2x2 = video_2x2.reshape([num_frames, 2 * 2 * 32, 3])
        video_3x3 = video_3x3.reshape([num_frames, 3 * 3 * 32, 3])
        video_4x4 = video_4x4.reshape([num_frames, 4 * 4 * 32, 3])
        video_5x5 = video_5x5.reshape([num_frames, 5 * 5 * 32, 3])
        # plt.imshow(video[:300].astype('uint8'))
        # plt.show(bbox_inches='tight', pad_inches=0.0)

        #
        # interpolation
        #
        num_frames_down = round(num_frames / video_fps * 30 / 1.5)
        num_frames_down = int(num_frames_down)
        num_frames_up = round(num_frames / video_fps * 30 * 1.5)
        num_frames_up = int(num_frames_up)
        num_frames_ori = round(num_frames / video_fps * 30)
        num_frames_ori = int(num_frames_ori)

        # num_frames = int(num_frames)
        video_fps = 30.0
        video_1x1_down = cv2.resize(video_1x1, (1 * 1 * 32, num_frames_down), interpolation=cv2.INTER_CUBIC)
        video_2x2_down = cv2.resize(video_2x2, (2 * 2 * 32, num_frames_down), interpolation=cv2.INTER_CUBIC)
        video_1x1_up = cv2.resize(video_1x1, (1 * 1 * 32, num_frames_up), interpolation=cv2.INTER_CUBIC)
        video_2x2_up = cv2.resize(video_2x2, (2 * 2 * 32, num_frames_up), interpolation=cv2.INTER_CUBIC)
        video_1x1_ori = cv2.resize(video_1x1, (1 * 1 * 32, num_frames_ori), interpolation=cv2.INTER_CUBIC)
        video_2x2_ori = cv2.resize(video_2x2, (2 * 2 * 32, num_frames_ori), interpolation=cv2.INTER_CUBIC)

        video_3x3_down = cv2.resize(video_3x3, (3 * 3 * 32, num_frames_down), interpolation=cv2.INTER_CUBIC)
        video_4x4_down = cv2.resize(video_4x4, (4 * 4 * 32, num_frames_down), interpolation=cv2.INTER_CUBIC)
        video_5x5_down = cv2.resize(video_5x5, (5 * 5 * 32, num_frames_down), interpolation=cv2.INTER_CUBIC)
        video_3x3_up = cv2.resize(video_3x3, (3 * 3 * 32, num_frames_up), interpolation=cv2.INTER_CUBIC)
        video_4x4_up = cv2.resize(video_4x4, (4 * 4 * 32, num_frames_up), interpolation=cv2.INTER_CUBIC)
        video_5x5_up = cv2.resize(video_5x5, (5 * 5 * 32, num_frames_up), interpolation=cv2.INTER_CUBIC)
        video_3x3_ori = cv2.resize(video_3x3, (3 * 3 * 32, num_frames_ori), interpolation=cv2.INTER_CUBIC)
        video_4x4_ori = cv2.resize(video_4x4, (4 * 4 * 32, num_frames_ori), interpolation=cv2.INTER_CUBIC)
        video_5x5_ori = cv2.resize(video_5x5, (5 * 5 * 32, num_frames_ori), interpolation=cv2.INTER_CUBIC)
        # print(video.shape)
        # exit(0)
        # plt.imshow(video[:300].astype('uint8'))
        # plt.show(bbox_inches='tight', pad_inches=0.0)
        # exit(0)
        # print(video.shape)  # (1055, 20, 3)
        # exit(0)

        video_1x1_down = video_1x1_down.reshape([num_frames_down, 1 * 1 * 32, 3])
        video_2x2_down = video_2x2_down.reshape([num_frames_down, 2 * 2 * 32, 3])
        video_1x1_up = video_1x1_up.reshape([num_frames_up, 1 * 1 * 32, 3])
        video_2x2_up = video_2x2_up.reshape([num_frames_up, 2 * 2 * 32, 3])
        video_1x1_ori = video_1x1_ori.reshape([num_frames_ori, 1 * 1 * 32, 3])
        video_2x2_ori = video_2x2_ori.reshape([num_frames_ori, 2 * 2 * 32, 3])
        video_3x3_down = video_3x3_down.reshape([num_frames_down, 3 * 3 * 32, 3])
        video_4x4_down = video_4x4_down.reshape([num_frames_down, 4 * 4 * 32, 3])
        video_5x5_down = video_5x5_down.reshape([num_frames_down, 5 * 5 * 32, 3])
        video_3x3_up = video_3x3_up.reshape([num_frames_up, 3 * 3 * 32, 3])
        video_4x4_up = video_4x4_up.reshape([num_frames_up, 4 * 4 * 32, 3])
        video_5x5_up = video_5x5_up.reshape([num_frames_up, 5 * 5 * 32, 3])
        video_3x3_ori = video_3x3_ori.reshape([num_frames_ori, 3 * 3 * 32, 3])
        video_4x4_ori = video_4x4_ori.reshape([num_frames_ori, 4 * 4 * 32, 3])
        video_5x5_ori = video_5x5_ori.reshape([num_frames_ori, 5 * 5 * 32, 3])
        print('num_frames = {}'.format(num_frames))
        print('num_frames_down = {}'.format(num_frames_down))
        print('num_frames_up = {}'.format(num_frames_up))
        print('num_frames_ori = {}'.format(num_frames_ori))

        if which_dataset in ['mahnob']:
            fs_org_sig = pulse_fps
        else:
            fs_org_sig = len(gtTrace) / (num_frames / video_fps)
        print('fs_org_sig: {}'.format(fs_org_sig))

        cleanTrace = butter_bandpass_filter(
            gtTrace,
            low_cut=0.6,
            high_cut=3.0,
            fs=fs_org_sig,
            order=4,
        )
        # cleanTrace = gtTrace
        # plt.plot(cleanTrace[:300])
        # plt.show()
        cleanTrace_down = spline_interpolate(sig=cleanTrace, target_len=num_frames_down)
        cleanTrace_up = spline_interpolate(sig=cleanTrace, target_len=num_frames_up)
        cleanTrace_ori = spline_interpolate(sig=cleanTrace, target_len=num_frames_ori)

        gtTrace_down = spline_interpolate(sig=gtTrace, target_len=num_frames_down)
        gtTrace_up = spline_interpolate(sig=gtTrace, target_len=num_frames_up)
        gtTrace_ori = spline_interpolate(sig=gtTrace, target_len=num_frames_ori)
        #
        # plt.plot(cleanTrace[:300])
        # plt.show()
        # exit(0)

        # exit(0)

        video_1x1_down = np.array(video_1x1_down, dtype='float32')
        video_2x2_down = np.array(video_2x2_down, dtype='float32')
        video_1x1_up = np.array(video_1x1_up, dtype='float32')
        video_2x2_up = np.array(video_2x2_up, dtype='float32')
        video_1x1_ori = np.array(video_1x1_ori, dtype='float32')
        video_2x2_ori = np.array(video_2x2_ori, dtype='float32')
        video_3x3_down = np.array(video_3x3_down, dtype='float32')
        video_4x4_down = np.array(video_4x4_down, dtype='float32')
        video_5x5_down = np.array(video_5x5_down, dtype='float32')
        video_3x3_up = np.array(video_3x3_up, dtype='float32')
        video_4x4_up = np.array(video_4x4_up, dtype='float32')
        video_5x5_up = np.array(video_5x5_up, dtype='float32')
        video_3x3_ori = np.array(video_3x3_ori, dtype='float32')
        video_4x4_ori = np.array(video_4x4_ori, dtype='float32')
        video_5x5_ori = np.array(video_5x5_ori, dtype='float32')

        print('video.shape: {}'.format(video_1x1.shape))
        print('video.shape: {}'.format(video_2x2.shape))
        # print('video.shape: {}'.format(video_3x3.shape))
        # print('video.shape: {}'.format(video_4x4.shape))
        # print('video.shape: {}'.format(video_5x5.shape))
        # save_data(
        #     save_dir=output_video_dir,
        #     video_1x1=video_1x1,
        #     video_2x2=video_2x2,
        #     # video_3x3=video_3x3,
        #     # video_4x4=video_4x4,
        #     # video_5x5=video_5x5,
        #     bvp=cleanTrace,
        #     # hr=gtHR,
        #     video_fps=video_fps,
        #     clip_length=WIN_LEN,
        #     step_size=STEP_SIZE,        joblib.dump(video_1x1_down, 'down.pkl')

        #     which_skin_threshold=which_skin_threshold,
        # )

        #
        # save data
        #
        joblib.dump(video_1x1_down, os.path.join(output_video_dir, 'video_1x1_down.pkl'))
        joblib.dump(video_2x2_down, os.path.join(output_video_dir, 'video_2x2_down.pkl'))
        joblib.dump(video_1x1_up, os.path.join(output_video_dir, 'video_1x1_up.pkl'))
        joblib.dump(video_2x2_up, os.path.join(output_video_dir, 'video_2x2_up.pkl'))
        joblib.dump(video_1x1_ori, os.path.join(output_video_dir, 'video_1x1_ori.pkl'))
        joblib.dump(video_2x2_ori, os.path.join(output_video_dir, 'video_2x2_ori.pkl'))

        joblib.dump(video_3x3_down, os.path.join(output_video_dir, 'video_3x3_down.pkl'))
        joblib.dump(video_4x4_down, os.path.join(output_video_dir, 'video_4x4_down.pkl'))
        joblib.dump(video_5x5_down, os.path.join(output_video_dir, 'video_5x5_down.pkl'))
        joblib.dump(video_3x3_up, os.path.join(output_video_dir, 'video_3x3_up.pkl'))
        joblib.dump(video_4x4_up, os.path.join(output_video_dir, 'video_4x4_up.pkl'))
        joblib.dump(video_5x5_up, os.path.join(output_video_dir, 'video_5x5_up.pkl'))
        joblib.dump(video_3x3_ori, os.path.join(output_video_dir, 'video_3x3_ori.pkl'))
        joblib.dump(video_4x4_ori, os.path.join(output_video_dir, 'video_4x4_ori.pkl'))
        joblib.dump(video_5x5_ori, os.path.join(output_video_dir, 'video_5x5_ori.pkl'))

        joblib.dump(gtTrace_down, os.path.join(output_video_dir, 'gtTrace_down.pkl'))
        joblib.dump(gtTrace_up, os.path.join(output_video_dir, 'gtTrace_up.pkl'))
        joblib.dump(gtTrace_ori, os.path.join(output_video_dir, 'gtTrace_ori.pkl'))

        joblib.dump(cleanTrace_down, os.path.join(output_video_dir, 'cleanTrace_down.pkl'))
        joblib.dump(cleanTrace_up, os.path.join(output_video_dir, 'cleanTrace_up.pkl'))
        joblib.dump(cleanTrace_ori, os.path.join(output_video_dir, 'cleanTrace_ori.pkl'))

        cv2.imwrite(os.path.join(output_video_dir, 'video_1x1_down.png'), video_1x1_down[:, :, ::-1], )
        cv2.imwrite(os.path.join(output_video_dir, 'video_2x2_down.png'), video_2x2_down[:, :, ::-1], )
        cv2.imwrite(os.path.join(output_video_dir, 'video_1x1_up.png'), video_1x1_up[:, :, ::-1], )
        cv2.imwrite(os.path.join(output_video_dir, 'video_2x2_up.png'), video_2x2_up[:, :, ::-1], )
        cv2.imwrite(os.path.join(output_video_dir, 'video_1x1_ori.png'), video_1x1_ori[:, :, ::-1], )
        cv2.imwrite(os.path.join(output_video_dir, 'video_2x2_ori.png'), video_2x2_ori[:, :, ::-1], )

        cv2.imwrite(os.path.join(output_video_dir, 'video_3x3_down.png'), video_3x3_down[:, :, ::-1], )
        cv2.imwrite(os.path.join(output_video_dir, 'video_4x4_down.png'), video_4x4_down[:, :, ::-1], )
        cv2.imwrite(os.path.join(output_video_dir, 'video_5x5_down.png'), video_5x5_down[:, :, ::-1], )
        cv2.imwrite(os.path.join(output_video_dir, 'video_3x3_up.png'), video_3x3_up[:, :, ::-1], )
        cv2.imwrite(os.path.join(output_video_dir, 'video_4x4_up.png'), video_4x4_up[:, :, ::-1], )
        cv2.imwrite(os.path.join(output_video_dir, 'video_5x5_up.png'), video_5x5_up[:, :, ::-1], )
        cv2.imwrite(os.path.join(output_video_dir, 'video_3x3_ori.png'), video_3x3_ori[:, :, ::-1], )
        cv2.imwrite(os.path.join(output_video_dir, 'video_4x4_ori.png'), video_4x4_ori[:, :, ::-1], )
        cv2.imwrite(os.path.join(output_video_dir, 'video_5x5_ori.png'), video_5x5_ori[:, :, ::-1], )


def main():
    for which_skin_threshold in [0, ]:
        gen_dataset(
            which_dataset=which_dataset,
            which_skin_threshold=which_skin_threshold,
        )


if __name__ == '__main__':
    main()
