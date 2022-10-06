import gc
import os
import joblib
from torch.utils.data import DataLoader, Dataset

from config.parameters import OUTPUT_DIR
from datasets import pure
from datasets import ubfc2
from datasets import vipl
from datasets import mahnob
from preprocess.gen_STmap import image_to_STmap_68
from tools.io_tools import mkdir_if_missing
from tools.video_tools import load_video_bgr, load_half_video_bgr

use_xxx = [True, True, True, True, True]

which_dataset = 'pure'
NUM_WORKERS = 1

if which_dataset == 'ubfc2':
    video_path_list = ubfc2.UBFC2.video_path_list
elif which_dataset == 'pure':
    video_path_list = pure.Pure.video_dir_list
elif which_dataset == 'vipl':
    video_path_list = vipl.Vipl.video_path_list
elif which_dataset == 'mahnob':
    video_path_list = mahnob.Mahnob.video_path_list
else:
    raise Exception

#
#
#
load_dir = os.path.join(
    OUTPUT_DIR,
    'preprocess_landmark',
    which_dataset,
)
output_dir = os.path.join(
    OUTPUT_DIR,
    'preprocess_crop',
    which_dataset,
)
print('which_dataset = "{}"'.format(which_dataset))
print('load_dir = "{}"'.format(load_dir))
print('output_dir = "{}"'.format(output_dir))


#
# load landmark result
#
def get_landmarks_list(num_frames: int, load_video_dir: str):
    landmarks_list = []

    for f_idx in range(num_frames):
        load_video_frame_path = os.path.join(
            load_video_dir,
            str(f_idx)
        )
        if os.path.isfile(load_video_frame_path):
            with open(load_video_frame_path, 'rb') as f:
                landmarks = joblib.load(f)
                f.close()
        else:
            landmarks = None

        landmarks_list.append(landmarks)

    return landmarks_list


class TempDataset(Dataset):
    def __init__(
            self,
            video,
            skin_threshold,
            load_video_dir: str,
            output_video_dir: str,
    ):
        super(TempDataset, self).__init__()
        self.video = video
        self.load_video_dir = load_video_dir
        self.output_video_dir = output_video_dir

        # get landmarks_list
        self.num_frames = len(video)
        self.skin_threshold = skin_threshold
        self.landmarks_list = get_landmarks_list(load_video_dir=load_video_dir, num_frames=self.num_frames)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        image = self.video[index]
        landmarks = self.landmarks_list[index]
        if landmarks is None:
            print('Continue: load_video_dir = "{}", index = {}'.format(self.load_video_dir, index))
            return 0
        else:
            landmarks = landmarks.astype('int32')
            if self.skin_threshold > 0:
                flag_segment = True
            else:
                flag_segment = False
            data = {}
            for i in range(1, 5 + 1):
                if not use_xxx[i - 1]:
                    continue
                bgr_map = image_to_STmap_68(
                    image,
                    landmarks,
                    flag_plot=True,
                    flag_segment=flag_segment,
                    skin_threshold=self.skin_threshold,
                    roi_x=i,
                    roi_y=i,
                )
                data['{}x{}'.format(i, i)] = bgr_map[:, ::-1]

            rgb_path = os.path.join(
                self.output_video_dir,
                '{}_{}.pkl'.format(index, self.skin_threshold)
            )
            joblib.dump(data, rgb_path)

        return 0


#
#
#
def crop_face():
    # ubfc 36
    for v_idx, video_path in enumerate(video_path_list):
        # if not (303 <= v_idx <= 400):
        #     continue
        print('Process: v_idx = {}, video_path = "{}"'.format(v_idx, video_path))

        load_video_dir = os.path.join(
            load_dir,
            str(v_idx),
        )
        output_video_dir = os.path.join(
            output_dir,
            str(v_idx),
        )
        mkdir_if_missing(output_video_dir)
        print('load_video_dir = "{}"'.format(load_video_dir))
        print('output_video_dir = "{}"'.format(output_video_dir))

        if which_dataset in ['pure']:
            video = pure.load_video_bgr(video_path)
        elif which_dataset in ['mahnob']:
            video = load_half_video_bgr(video_path)
        else:
            video = load_video_bgr(video_path)

        # crop

        temp_dataset = TempDataset(
            video=video,
            skin_threshold=0,
            load_video_dir=load_video_dir,
            output_video_dir=output_video_dir,
        )
        temp_loader = DataLoader(
            temp_dataset,
            batch_size=64,
            drop_last=False,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        for batch in temp_loader:
            pass

        del video
        gc.collect()
        exit(0)


if __name__ == '__main__':
    crop_face()

