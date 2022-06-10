import os
import json
import numpy as np
import cv2

from config.parameters import PURE_DIR

# database root directory
ROOT_DIR = PURE_DIR


def _check_file(root_dir: str):
    # Return a list of video paths and a list of json file paths
    # json file list
    json_name_list = [_ for _ in os.listdir(root_dir) if '.json' in _]
    json_name_list.sort()
    # folder name list
    folder_name_list = [_[:-1 * len('.json')] for _ in json_name_list]
    assert len(folder_name_list) == 59
    # json path list
    json_path_list = [os.path.join(root_dir, _) for _ in json_name_list]
    # folder path list
    folder_path_list = [os.path.join(root_dir, _) for _ in folder_name_list]
    return folder_path_list, json_path_list


def load_video_bgr(video_dir: str) -> list:
    pic_names = [_ for _ in os.listdir(video_dir) if '.png' in _]
    pic_names.sort(key=lambda _: int(_[len('Image'):-1 * len('.png')]))

    video = []
    for pic_name in pic_names:
        pic_path = os.path.join(video_dir, pic_name)
        pic = cv2.imread(pic_path)
        video.append(pic)

    return video


def num_frames_video(video_dir: str) -> int:
    cnt = 0
    for _ in os.listdir(video_dir):
        if '.png' in _:
            cnt += 1
    return cnt


def read_signal_file(filename):
    """ Load BVP signal.
        Must return a 1-dim (row array) signal
    """
    bvp = []
    hr = []
    with open(filename) as json_file:
        json_data = json.load(json_file)
        for p in json_data['/FullPackage']:
            bvp.append(p['Value']['waveform'])
            hr.append(p['Value']['pulseRate'])

    bvp = np.array(bvp, dtype=np.float32).reshape([-1])
    hr = np.array(hr, dtype=np.float32).reshape([-1])
    return bvp, hr


def get_p_idx_list(json_path_list):
    p_idx_list = []
    for json_path in json_path_list:
        p_idx_list.append(
            int(json_path[json_path.rindex('/') + 1:json_path.rindex('-')])
        )
    return p_idx_list


class Pure:
    video_dir_list, json_path_list = _check_file(ROOT_DIR)
    p_idx_list = get_p_idx_list(json_path_list)
    p_idx_set = list(set(p_idx_list))


def demo():
    print('-' * 70)
    print('PURE::')
    print('video_dir_list:')
    print(Pure.video_dir_list)
    print('len(video_dir_list) = {}'.format(len(Pure.video_dir_list)))
    print('json_path_list:')
    print(Pure.json_path_list)
    print('len(json_path_list) = {}'.format(len(Pure.json_path_list)))
    print('-' * 70)


if __name__ == '__main__':
    demo()

"""
PURE contains 60 RGB videos from 10 subjects with 6 different activities (sitting
still, talking, four variations of rotating and moving head), which were recorded
using an eco274CVGE camera at 30 fps and a resolution of 640 x 480. The BVP signals
was collected using CMS50E at 60 fps. 

The BVP signals are reduced to 30 fps with spline interpolation to align with the videos.

[1] Dual-GAN: Joint BVP and Noise Modeling for Remote Physiological Measurement.
"""
