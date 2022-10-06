import os
import time

#
# todo
#
DBS_DIR = '/path/to/database/dir_name'
OUTPUT_DIR = '/path/to/output/dir_name'
#
# database dir
#
UBFC_rPPG_DATASET_2_DIR = os.path.join(DBS_DIR, 'UBFC_rPPG', 'DATASET_2')
PURE_DIR = os.path.join(DBS_DIR, 'PURE')
VIPL_DIR = os.path.join(DBS_DIR, 'VIPL_HR_V1')
MAHNOB_HCI_DIR = os.path.join(DBS_DIR, 'MAHNOB_HCI', 'Sessions')
#
# config
#
CFG = {
    'MODEL_SAVE_DIR': os.path.join(OUTPUT_DIR, 'model_save') + '_' + time.strftime('%Y-%m-%d-%H-%M-%S'),
    'SEED': 1024,
    'DEVICE_ID': '4',
    'PRE_TRAIN_CHOICE': '',
    'PRE_TRAIN_PATH': '',
    'train_txt_path': os.path.join(OUTPUT_DIR, 'train.txt'),
    'valid_txt_path': os.path.join(OUTPUT_DIR, 'valid.txt'),
    'EPOCH_NUM': 40,
    'LOG_PERIOD': 50,
    'SAVE_PERIOD': 1,
    'EVAL_PERIOD': 1,
    'lr': 0.0001,
    'TRAIN_BATCH_SIZE': 32,
    'VALID_BATCH_SIZE': 32,
}

if __name__ == '__main__':
    assert os.path.isdir(DBS_DIR)
    assert os.path.isdir(UBFC_rPPG_DATASET_2_DIR)
    assert os.path.isdir(PURE_DIR)
    assert os.path.isdir(MAHNOB_HCI_DIR)
    assert os.path.isdir(VIPL_DIR)
    assert os.path.isdir(OUTPUT_DIR)
