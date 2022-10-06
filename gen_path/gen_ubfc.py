import os

from config.parameters import OUTPUT_DIR

ubfc_dir = '/path/to/output/preprocess_data/ubfc2/'

train_txt_path = os.path.join(OUTPUT_DIR, 'train.txt')
valid_txt_path = os.path.join(OUTPUT_DIR, 'valid.txt')

with open(train_txt_path, 'w') as f:
    for i in range(30):
        train_path = os.path.join(ubfc_dir, str(i), 'cleanTrace_ori.pkl')
        f.write(train_path + '\n')

    f.close()

with open(valid_txt_path, 'w') as f:
    for i in range(30, 42):
        valid_path = os.path.join(ubfc_dir, str(i), 'cleanTrace_ori.pkl')
        f.write(valid_path + '\n')
    f.close()
