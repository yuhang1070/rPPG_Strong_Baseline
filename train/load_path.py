from config.parameters import OUTPUT_DIR
from tools.load_tools import load_path_list
import os

TXT_DIR = os.path.join(OUTPUT_DIR, 'txt_dir')


def load_train_valid(db_name, v_idx=None):
    if db_name in [
        'ubfc2', 'pure', 'pure_ubfc2',
    ]:
        train_txt_path_list = [
            os.path.join(TXT_DIR, db_name, 'train.txt'),
        ]
        valid_txt_path_list = [
            os.path.join(TXT_DIR, db_name, 'valid.txt'),
        ]
    elif db_name == 'vipl':
        if v_idx not in list(range(5)):
            raise Exception
        train_txt_path_list = []
        for _ in range(5):
            if _ != v_idx:
                train_txt_path_list.append(
                    os.path.join(TXT_DIR, db_name, '{}.txt'.format(_)),
                )
        valid_txt_path_list = [
            os.path.join(TXT_DIR, db_name, '{}.txt'.format(v_idx)),
        ]
    else:
        raise Exception('Unknown dataset = {}'.format(db_name))

    train_path_list = []
    for _ in train_txt_path_list:
        train_path_list += load_path_list(txt_path=_, )
    valid_path_list = []
    for _ in valid_txt_path_list:
        valid_path_list += load_path_list(txt_path=_, )
    if db_name in ['vipl']:
        assert len(train_path_list) + len(valid_path_list) == 2378

    return train_path_list, valid_path_list


def load_all_train_valid(db_list=None):
    train_txt_path_list = []
    if 'ubfc2' in db_list:
        train_txt_path_list.append(
            os.path.join(TXT_DIR, 'ubfc2', 'train.txt'),
        )
        train_txt_path_list.append(
            os.path.join(TXT_DIR, 'ubfc2', 'valid.txt'),
        )
    if 'pure' in db_list:
        train_txt_path_list.append(
            os.path.join(TXT_DIR, 'pure', 'train.txt'),
        )
        train_txt_path_list.append(
            os.path.join(TXT_DIR, 'pure', 'valid.txt'),
        )
    if 'vipl' in db_list:
        for _ in range(5):
            train_txt_path_list.append(
                os.path.join(TXT_DIR, 'vipl', '{}.txt'.format(_)),
            )

    if 'mahnob' in db_list:
        train_txt_path_list.append(
            os.path.join(TXT_DIR, 'mahnob', 'all.txt'),
        )

    train_path_list = []
    for _ in train_txt_path_list:
        train_path_list += load_path_list(txt_path=_, )

    return train_path_list


def demo():
    p_l = load_all_train_valid(
        [
            'vipl',
            'pure',
            'ubfc2',
            'mahnob',
        ]
    )
    print(len(p_l))


if __name__ == '__main__':
    demo()
