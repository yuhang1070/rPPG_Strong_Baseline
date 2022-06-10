from config.parameters import UBFC_rPPG_DATASET_2_DIR

import os
import numpy as np

ROOT_DIR = UBFC_rPPG_DATASET_2_DIR


def _check_file(root_dir: str, folder_name_list: list, ):
    video_path_list = []
    gt_path_list = []
    for folder_name in folder_name_list:
        subject_dir = os.path.join(root_dir, folder_name)
        # txt path
        txt_path = os.path.join(subject_dir, 'ground_truth.txt')
        if not os.path.exists(txt_path):
            raise Exception('File Not Exist: `{}`'.format(txt_path))
        # avi path
        gt_path_list.append(txt_path)
        avi_path = os.path.join(subject_dir, 'vid.avi')
        if not os.path.exists(avi_path):
            raise Exception('File Not Exist: `{}`'.format(avi_path))
        video_path_list.append(avi_path)
    return video_path_list, gt_path_list


def read_signal_file(filepath: str):
    """ Load BVP signal.
        Must return a 1-dim (row array) signal
    """
    content = np.loadtxt(filepath, delimiter=None, dtype='float32')
    gtTrace = content[0, :]
    gtHR = content[1, :]
    gtTime = content[2, :]
    # SIG_SampleRate = np.round(1 / np.mean(np.diff(gtTime)))
    # SIG_SampleRate = int(np.round((len(gtTime) - 1) / (gtTime[-1] - gtTime[0])))
    return gtTrace, gtHR, gtTime


#
# UBFC2
#
class UBFC2:
    folder_name_list = ['subject1', 'subject3', 'subject4', 'subject5', 'subject8', 'subject9', 'subject10',
                        'subject11', 'subject12', 'subject13', 'subject14', 'subject15', 'subject16', 'subject17',
                        'subject18', 'subject20', 'subject22', 'subject23', 'subject24', 'subject25', 'subject26',
                        'subject27', 'subject30', 'subject31', 'subject32', 'subject33', 'subject34', 'subject35',
                        'subject36', 'subject37', 'subject38', 'subject39', 'subject40', 'subject41', 'subject42',
                        'subject43', 'subject44', 'subject45', 'subject46', 'subject47', 'subject48', 'subject49']
    assert len(folder_name_list) == 42

    video_path_list, gt_path_list = _check_file(ROOT_DIR, folder_name_list)


def demo():
    print('-' * 70)
    print('UBFC2::')
    print('folder_name_list:')
    print(UBFC2.folder_name_list)
    print('len(folder_name_list): {}'.format(len(UBFC2.folder_name_list)))
    print('video_path_list:')
    print(UBFC2.video_path_list)
    print('len(video_path_list): {}'.format(len(UBFC2.video_path_list)))
    print('GT_PATH_LIST:')
    print(UBFC2.gt_path_list)
    print('len(GT_PATH_LIST): {}'.format(len(UBFC2.gt_path_list)))
    print('-' * 70)


if __name__ == '__main__':
    demo()

"""
UBFC-rPPG contains 42 RGB videos containing sunlight and indoor illumination.
The videos were recorded with a Logitech C920 HD Pro webcam in a resolution of
640 x 480 and 30 fps. The ground-truth BVP signals and HR values were collected
by CMS50E.

[1] Dual-GAN: Joint BVP and Noise Modeling for Remote Physiological Measurement.
"""
