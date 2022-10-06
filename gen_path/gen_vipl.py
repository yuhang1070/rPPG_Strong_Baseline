import os
import numpy as np

from datasets import vipl
from config.parameters import OUTPUT_DIR

vipl_dir = '/path/to/output/preprocess_data/vipl/'

train_txt_path = os.path.join(OUTPUT_DIR, 'train.txt')
valid_txt_path = os.path.join(OUTPUT_DIR, 'valid.txt')
#

fold_content = vipl.Vipl.fold_content
fold_content = np.array(fold_content)
video_path_list = vipl.Vipl.video_path_list
train_sid_list = []
valid_sid_list = []
print('fold_content:', fold_content)

selected_idx = 0
other_idx_list = []
for _ in range(len(fold_content)):
    if _ != selected_idx:
        other_idx_list.append(_)

print('selected_idx:', selected_idx)
print('other_idx_list:', other_idx_list)
for _ in fold_content[selected_idx:selected_idx + 1]:
    for __ in _:
        valid_sid_list.append(__)

for _ in fold_content[other_idx_list]:
    for __ in _:
        train_sid_list.append(__)
train_sid_list.sort()
valid_sid_list.sort()
print('train_sid_list: {}'.format(train_sid_list))
print('valid_sid_list: {}'.format(valid_sid_list))

for v_idx, video_path in enumerate(video_path_list):
    sid = vipl.Vipl.sid_list[v_idx]
    print('v_idx = {}, video_path = {}, sid = {}'.format(v_idx, video_path, sid))

    ori_path = os.path.join(
        vipl_dir, str(v_idx), 'cleanTrace_ori.pkl'
    )
    up_path = os.path.join(
        vipl_dir, str(v_idx), 'cleanTrace_up.pkl'
    )
    down_path = os.path.join(
        vipl_dir, str(v_idx), 'cleanTrace_down.pkl'
    )
    use_train = sid in train_sid_list
    if use_train:
        txt_path = train_txt_path
    else:
        txt_path = valid_txt_path

    with open(txt_path, 'a') as f:
        f.write(ori_path + '\n')
        # if use_train:
        #     f.write(up_path + '\n')
        #     f.write(down_path + '\n')
