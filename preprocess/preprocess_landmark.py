from attachment.pytorch_face_landmark.common.utils import BBox
from attachment.pytorch_face_landmark.models import MobileNet_GDConv
from attachment.pytorch_face_landmark.models.mobilefacenet import MobileFaceNet
from attachment.pytorch_face_landmark.models.pfld_compressed import PFLDInference

from datasets import pure
from datasets import mahnob
from datasets import ubfc2
from datasets import vipl

from tools.io_tools import mkdir_if_missing
from tools.net_tools import remove_prefix, check_keys
from config.parameters import OUTPUT_DIR
from tools.video_tools import load_video_bgr, load_half_video_bgr

import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import joblib
import gc

#
# global config
#

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
torch.set_grad_enabled(False)
mean = np.asarray([0.485, 0.456, 0.406])
std = np.asarray([0.229, 0.224, 0.225])
#
#
# landmark face
#
#
which_model = 'MobileFaceNet'

which_dataset = 'vipl'

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

load_dir = os.path.join(
    OUTPUT_DIR,
    'preprocess_detect',
    which_dataset,
)
output_dir = os.path.join(
    OUTPUT_DIR,
    'preprocess_landmark',
    which_dataset,
)


#
#
#


def load_model():
    device = 'cuda'
    if which_model == 'MobileFaceNet':
        model = MobileFaceNet([112, 112], 136).to(device)
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'attachment',
            'pytorch_face_landmark',
            'checkpoint',
            'mobilefacenet_model_best.pth.tar'
        )
    elif which_model == 'PFLD':
        model = PFLDInference()
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'attachment',
            'pytorch_face_landmark',
            'checkpoint',
            'pfld_model_best.pth.tar'
        )
    elif which_model == 'MobileNet':
        model = MobileNet_GDConv(136)
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'attachment',
            'pytorch_face_landmark',
            'checkpoint',
            'mobilenet_224_model_best_gdconv_external.pth.tar'
        )
    else:
        raise Exception
    map_location = device

    pretrained_dict = torch.load(model_path, map_location=map_location)
    # model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.to(device)

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict)
    return model


def get_faces_and_bboxes(height: int, width: int, num_frames: int, load_video_dir: str):
    new_bboxes = []
    faces = []
    borders = []
    for f_idx in range(num_frames):
        load_video_frame_path = os.path.join(
            load_video_dir,
            str(f_idx)
        )
        if not os.path.isfile(load_video_frame_path):
            new_bboxes.append(None)
            faces.append([0, 0, 0, 0, 0])
            borders.append([0, 0, 0, 0])
        else:
            face = joblib.load(load_video_frame_path)
            # process face
            x1 = face[0]
            y1 = face[1]
            x2 = face[2]
            y2 = face[3]
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            size = int(min([w, h]) * 1.2)
            cx = x1 + w // 2
            cy = y1 + h // 2
            x1 = cx - size // 2
            x2 = x1 + size
            y1 = cy - size // 2
            y2 = y1 + size

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)
            new_bbox_ = list(map(int, [x1, x2, y1, y2]))
            new_bbox_ = BBox(new_bbox_)

            # append
            faces.append(face)
            borders.append([dx, dy, edx, edy])
            new_bboxes.append(new_bbox_)

    return faces, borders, new_bboxes


# landmark
class LandmarkDataset(Dataset):
    def __init__(self, video_, bboxes_, borders_):
        super(LandmarkDataset, self).__init__()
        self.video = video_
        self.bboxes = bboxes_
        self.borders = borders_
        self.num_frames = len(self.video)
        self.out_size = 112
        if which_model == 'MobileNet':
            self.out_size = 224

        self.device = 'cuda'

    def __len__(self):
        return self.num_frames

    def __getitem__(self, frame_idx_):
        img = self.video[frame_idx_]

        bbox = self.bboxes[frame_idx_]
        if bbox is None:
            return torch.zeros((3, self.out_size, self.out_size), dtype=torch.float32, device=self.device)

        border = self.borders[frame_idx_]
        dx, dy, edx, edy = border

        # crop
        cropped = img[bbox.top:bbox.bottom, bbox.left:bbox.right]
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
        cropped_face = cv2.resize(cropped, (self.out_size, self.out_size))

        if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
            raise Exception

        test_face = cropped_face
        test_face = test_face / 255.0
        if which_model == 'MobileNet':
            test_face = (test_face - mean) / std
        test_face = test_face.transpose((2, 0, 1))
        ipt = torch.from_numpy(test_face).float().to(self.device)
        ipt = torch.autograd.Variable(ipt)

        return ipt


def landmark_face():
    # model
    model = load_model()

    for v_idx, video_path in enumerate(video_path_list):
        if v_idx not in [237,
                         248,
                         348,
                         503,
                         1114, ]:
            continue
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

        # get new bboxes
        image = video[0]
        height, width, _ = image.shape
        num_frames = len(video)
        faces, borders, new_bboxes = get_faces_and_bboxes(
            height=height,
            width=width,
            num_frames=num_frames,
            load_video_dir=load_video_dir,
        )

        # landmark
        dataset = LandmarkDataset(video_=video, bboxes_=new_bboxes, borders_=borders)
        data_loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=512,
            drop_last=False,
        )
        start_idx = 0

        for b_idx, batch in enumerate(data_loader):
            batch_size = batch.shape[0]
            if which_model == 'MobileFaceNet':
                tmp_landmarks = model(batch)[0].cpu().data.numpy()
            else:
                tmp_landmarks = model(batch).cpu().data.numpy()
            tmp_landmarks = tmp_landmarks.reshape(batch_size, -1, 2)
            for l_idx in range(batch_size):
                new_bbox = new_bboxes[start_idx]
                if new_bbox is None:
                    print('Continue: v_idx = {}, start_idx = {}'.format(v_idx, start_idx))
                    start_idx += 1
                    continue
                else:
                    landmark = tmp_landmarks[l_idx]
                    landmark = new_bbox.reprojectLandmark(landmark)

                    dump_video_frame_path = os.path.join(
                        output_video_dir,
                        str(start_idx)
                    )
                    joblib.dump(landmark, dump_video_frame_path)
                    start_idx += 1

        del video
        gc.collect()


if __name__ == '__main__':
    landmark_face()
