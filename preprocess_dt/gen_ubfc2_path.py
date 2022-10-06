from config.parameters import OUTPUT_DIR
from datasets import ubfc2
import numpy as np
import os

"""
Following the protocol in [31], the videos of first 30 subjects are used for training, 
and the videos of remaining 12 subjects are used for testing.

For all the experiments, the length of each video chip is set to 256 frames,
and the step between clips is 10 frames.

In all experiments, we compute HR, HRV, and RF based on the average duration
of adjacent BVP signals peaks.
"""


which_dataset = 'ubfc2'
# if which_dataset == ''

load_dir = os.path.join(
    OUTPUT_DIR,
    'preprocess_crop',
    which_dataset,
)

output_dir = os.path.join(
    OUTPUT_DIR,
    'preprocess_dt',
    which_dataset,
)

video_path_list = ubfc2.UBFC2.video_path_list

train_txt_path = os.path.join(OUTPUT_DIR, 'train.txt')
valid_txt_path = os.path.join(OUTPUT_DIR, 'valid.txt')

# if os.path.isfile(train_txt_path):
#     with open(train_txt_path, 'w') as f:
#         f.close()
if os.path.isfile(valid_txt_path):
    with open(valid_txt_path, 'w') as f:
        f.close()

for v_idx, video_path in enumerate(video_path_list):
    print('v_idx = {}, video_path = {}'.format(v_idx, video_path))
    use_train = v_idx < 30
    # use_train = False
    if use_train:
        txt_path = train_txt_path
    else:
        txt_path = valid_txt_path
    load_video_dir = os.path.join(
        load_dir,
        str(v_idx),
    )
    output_video_dir = os.path.join(
        output_dir,
        str(v_idx),
    )
    print(output_video_dir)

    if os.path.isfile(txt_path):
        mode = 'a'
    else:
        mode = 'w'

    if use_train:
        # train
        pkl_name_list = os.listdir(output_video_dir)
        pkl_name_list.sort()
        with open(txt_path, mode) as f:
            for pkl_name in pkl_name_list:
                pkl_path = os.path.join(output_video_dir, pkl_name) + '\n'
                f.write(pkl_path)

        output_video_dir_up = output_video_dir.replace('preprocess_dt', 'preprocess_dt_up')
        pkl_name_list = os.listdir(output_video_dir_up)
        pkl_name_list.sort()
        with open(txt_path, mode) as f:
            for pkl_name in pkl_name_list:
                pkl_path = os.path.join(output_video_dir_up, pkl_name) + '\n'
                f.write(pkl_path)

        output_video_dir_down = output_video_dir.replace('preprocess_dt', 'preprocess_dt_down')
        pkl_name_list = os.listdir(output_video_dir_down)
        pkl_name_list.sort()
        with open(txt_path, mode) as f:
            for pkl_name in pkl_name_list:
                pkl_path = os.path.join(output_video_dir_down, pkl_name) + '\n'
                f.write(pkl_path)
    else:
        # valid
        pkl_name_list = os.listdir(output_video_dir)
        pkl_name_list.sort()
        with open(txt_path, mode) as f:
            for pkl_name in pkl_name_list:
                pkl_path = os.path.join(output_video_dir, pkl_name) + '\n'
                f.write(pkl_path)
