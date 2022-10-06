from config.parameters import OUTPUT_DIR
from datasets import pure
import numpy as np
import os

which_dataset = 'pure'


# if which_dataset == ''

def string_list_in_string(aList, aStr, ):
    # print('string_list_in_string')
    # print('aList', aList)
    for _ in aList:
        # print(_, aStr)
        if _ in aStr:
            # print('True')
            return True
    return False


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

video_path_list = pure.Pure.video_dir_list

train_txt_path = os.path.join(OUTPUT_DIR, 'train.txt')
valid_txt_path = os.path.join(OUTPUT_DIR, 'valid.txt')

if os.path.isfile(train_txt_path):
    with open(train_txt_path, 'w') as f:
        f.close()
if os.path.isfile(valid_txt_path):
    with open(valid_txt_path, 'w') as f:
        f.close()

valid_list = []
for v_idx in [2, 3, 10]:
    for s_idx in range(1, 6 + 1, 1):
        valid_list.append('{}-{:02}'.format(v_idx, s_idx))
print(valid_list)
for v_idx, video_path in enumerate(video_path_list):
    use_valid = string_list_in_string(valid_list, video_path)
    # use_valid = False
    if use_valid:
        txt_path = valid_txt_path
        print('valid: v_idx = {}, video_path = {}'.format(v_idx, video_path))
    else:
        txt_path = train_txt_path
        print('train: v_idx = {}, video_path = {}'.format(v_idx, video_path))

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

    if not use_valid:
        # train
        with open(txt_path, mode) as f:
            f.write(output_video_dir + '\n')
    else:
        # valid
        with open(txt_path, mode) as f:
            f.write(output_video_dir + '\n')
