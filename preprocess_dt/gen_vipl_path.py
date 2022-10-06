from config.parameters import OUTPUT_DIR
from datasets import vipl
import numpy as np
import os

which_dataset = 'vipl'

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

video_path_list = vipl.Vipl.video_path_list
fold_content = vipl.Vipl.fold_content
fold_content = np.array(fold_content)
train_sid_list = []
valid_sid_list = []

print('fold_content:\n', fold_content)
# print(len(fold_content)) # 5

selected_idx = 0
other_idx_list = []
for _ in range(len(fold_content)):
    if _ != selected_idx:
        other_idx_list.append(_)

print(selected_idx)
print(other_idx_list)
# print(fold_content[selected_idx:selected_idx+1])
for _ in fold_content[selected_idx:selected_idx+1]:
    for __ in _:
        valid_sid_list.append(__)

for _ in fold_content[other_idx_list]:
    for __ in _:
        train_sid_list.append(__)

train_sid_list.sort()
valid_sid_list.sort()
print('train_sid_list: {}'.format(train_sid_list))
print('valid_sid_list: {}'.format(valid_sid_list))

train_txt_path = os.path.join(OUTPUT_DIR, 'train.txt')
valid_txt_path = os.path.join(OUTPUT_DIR, 'valid.txt')

for v_idx, video_path in enumerate(video_path_list):
    sid = vipl.Vipl.sid_list[v_idx]
    print('v_idx = {}, video_path = {}, sid = {}'.format(v_idx, video_path, sid))
    # if v_idx in vipl.Vipl.black_vid_list:
    #     print('Continue')
    #     continue
    load_video_dir = os.path.join(
        load_dir,
        str(v_idx),
    )
    output_video_dir = os.path.join(
        output_dir,
        str(v_idx),
    )
    print(output_video_dir)

    pkl_name_list = os.listdir(output_video_dir)
    pkl_name_list.sort()

    if sid in train_sid_list:
        txt_path = train_txt_path
    else:
        txt_path = valid_txt_path
    if os.path.isfile(txt_path):
        mode = 'a'
    else:
        mode = 'w'
    with open(txt_path, mode) as f:
        for pkl_name in pkl_name_list:
            pkl_path = os.path.join(output_video_dir, pkl_name) + '\n'
            f.write(pkl_path)




