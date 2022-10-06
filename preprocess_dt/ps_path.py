from config.parameters import OUTPUT_DIR
from datasets import vipl

import os
import glob


def get_pkl_path_list(_dir_list):
    pkl_path_list = []
    for _dir in _dir_list:
        tmp_pkl_path_list = glob.glob(os.path.join(_dir, '*.pkl'))
        tmp_pkl_path_list.sort()
        pkl_path_list += tmp_pkl_path_list
    return pkl_path_list


def write_path_list(filepath, pkl_path_list, mode='w'):
    with open(filepath, mode=mode) as f:
        for pkl_path in pkl_path_list:
            f.write(pkl_path + '\n')


def split_train_test(lst: list, valid_part_id, part_num):
    lst_len = len(lst)
    part_len = lst_len // part_num
    if valid_part_id == part_num:
        idx1 = part_len * (valid_part_id - 1)
        train_part = lst[:idx1]
        valid_part = lst[idx1:]
    elif valid_part_id == 1:
        idx1 = part_len
        valid_part = lst[:idx1]
        train_part = lst[idx1:]
    else:
        idx1 = part_len * (valid_part_id - 1)
        idx2 = part_len * valid_part_id
        valid_part = lst[idx1:idx2]
        train_part = lst[:idx1] + lst[idx2:]
    # assert len(train_part) // len(valid_part) == (part_num - 1)
    return train_part, valid_part


def gen_path_txt():
    train_txt_path = os.path.join(OUTPUT_DIR, 'train.txt')
    valid_txt_path = os.path.join(OUTPUT_DIR, 'valid.txt')

    #
    # track1train
    #
    track1train_folder = os.path.join(OUTPUT_DIR, 'preprocess_dt', 'track1train')
    track1train_dirs = glob.glob(os.path.join(track1train_folder, '*'))
    track1train_dirs.sort(key=lambda _: int(_[_.rindex('/') + 1:]))

    #
    # vipl
    #
    vipl_folder = os.path.join(OUTPUT_DIR, 'preprocess_dt', 'vipl')
    vipl_dirs = glob.glob(os.path.join(vipl_folder, '*'))
    vipl_dirs.sort(key=lambda _: int(_[_.rindex('/') + 1:]))

    #
    # pure
    #
    pure_folder = os.path.join(OUTPUT_DIR, 'preprocess_dt', 'pure')
    pure_dirs = glob.glob(os.path.join(pure_folder, '*'))
    pure_dirs.sort(key=lambda _: int(_[_.rindex('/') + 1:]))

    #
    # ubfc2
    #
    ubfc2_folder = os.path.join(OUTPUT_DIR, 'preprocess_dt', 'ubfc2')
    ubfc2_dirs = glob.glob(os.path.join(ubfc2_folder, '*'))
    ubfc2_dirs.sort(key=lambda _: int(_[_.rindex('/') + 1:]))

    #
    # for train
    #
    write_path_list(
        filepath=train_txt_path,
        pkl_path_list=get_pkl_path_list(
            split_train_test(
                vipl_dirs,
                1,
                5
            )[0],
        ),
        mode='w'
    )

    write_path_list(
        filepath=valid_txt_path,
        pkl_path_list=get_pkl_path_list(
            split_train_test(
                vipl_dirs,
                1,
                5
            )[1],
        ),
        mode='w',
    )


if __name__ == '__main__':
    gen_path_txt()
