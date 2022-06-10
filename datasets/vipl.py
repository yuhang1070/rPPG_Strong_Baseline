import os
import numpy as np
from scipy.io import loadmat
from tools.video_tools import num_frames_video
from config.parameters import VIPL_DIR

# DIR to VIPL-HR database
ROOT_DIR = VIPL_DIR


# #################################################
# read file function
# #################################################
# read time.txt
def read_time(filepath: str) -> np.ndarray:
    gt_time = np.loadtxt(filepath, skiprows=0)
    return gt_time


# read gt_HR.csv
def read_gt_HR(filepath: str) -> np.ndarray:
    gt_HR = np.loadtxt(filepath, skiprows=1)
    return gt_HR


# read wave.csv
def read_wave(filepath: str) -> np.ndarray:
    bvp = np.loadtxt(filepath, skiprows=1)
    return bvp


# #################################################
# VIPL-HR
# #################################################
def _check_file(root_dir: str):
    # dir to data
    data_dir = os.path.join(root_dir, 'data')

    # person name list
    person_name_list = [_ for _ in os.listdir(data_dir) if 'p' in _]
    person_name_list.sort(key=lambda _: int(_[1:]))

    hr_path_list = []
    fps_list = []
    video_path_list = []
    wave_path_list = []
    sid_list = []

    for p_idx, person_name in enumerate(person_name_list):
        p_dir = os.path.join(data_dir, person_name)
        v_name_list = os.listdir(p_dir)
        v_name_list.sort()
        for v_name in v_name_list:
            v_dir = os.path.join(p_dir, v_name)
            s_name_list = os.listdir(v_dir)
            s_name_list.sort(key=lambda _: int(_[len('source'):]))
            for s_name in s_name_list:
                s_dir = os.path.join(v_dir, s_name)
                if s_name == 'source4':
                    # print('Continue NIR: {}'.format([person_name, v_name, s_name]))
                    continue
                hr_path = os.path.join(s_dir, 'gt_HR.csv')
                time_path = os.path.join(s_dir, 'time.txt')
                video_path = os.path.join(s_dir, 'video.avi')
                wave_path = os.path.join(s_dir, 'wave.csv')

                ex_hr = os.path.isfile(hr_path)
                ex_time = os.path.isfile(time_path)
                ex_video = os.path.isfile(video_path)
                ex_wave = os.path.isfile(wave_path)

                ex_all = ex_hr and ex_video and ex_wave
                if not ex_all:
                    print('Not complete: "{}"'.format(s_dir))
                    continue
                elif ex_all and not ex_time:
                    fps = 30.
                else:
                    time = read_time(time_path)
                    # nf = num_frames_video(video_path)
                    # fps_1 = nf / (time[-1] - time[0]) * 1000

                    fps = (len(time) - 1) / (time[-1] - time[0]) * 1000

                    # if fps_1 < 15:
                    #     print(fps_1, video_path)
                    #
                    #
                    # if nf != len(time):
                    #     print(nf)
                    #     print('fps_1: {}, fps: {}'.format(fps_1, fps))

                fps_list.append(fps)
                hr_path_list.append(hr_path)
                video_path_list.append(video_path)
                wave_path_list.append(wave_path)
                sid_list.append(p_idx)

    return hr_path_list, fps_list, video_path_list, wave_path_list, sid_list


def get_fold_content(fold_dir: str, use_sid=True):
    # sid: subject id
    fold_content = []

    for f_idx in range(1, 5 + 1):
        fold_path = os.path.join(fold_dir, 'fold{}.mat'.format(f_idx))
        fold = loadmat(fold_path)['fold{}'.format(f_idx)].reshape([-1])
        fold = np.array(fold, dtype=np.int32)
        if use_sid:
            fold -= 1
        fold_content.append(fold.tolist())
    return fold_content


class Vipl:
    hr_path_list, fps_list, video_path_list, wave_path_list, sid_list = _check_file(root_dir=ROOT_DIR)
    assert len(video_path_list) == 2378
    fold_split_dir = os.path.join(VIPL_DIR, 'fold_split')
    fold_content = get_fold_content(fold_split_dir, use_sid=True)





