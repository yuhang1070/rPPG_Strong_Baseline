from config.parameters import OUTPUT_DIR
import os

mahnob_dir = '/path/to/output/preprocess_data/mahnob/'

all_txt_path = os.path.join(OUTPUT_DIR, 'all.txt')

with open(all_txt_path, 'w') as f:
    for i in range(527):
        train_path = os.path.join(mahnob_dir, str(i), 'cleanTrace_ori.pkl')
        f.write(train_path + '\n')

    f.close()
