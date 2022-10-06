from attachment.pytorch_face_landmark.Retinaface.layers.functions.prior_box import PriorBox
from attachment.pytorch_face_landmark.Retinaface.utils.box_utils import decode
from attachment.pytorch_face_landmark.Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from attachment.pytorch_face_landmark.Retinaface.models.retinaface import RetinaFace

from datasets import pure
from datasets import ubfc2
from datasets import vipl
# from datasets import mahnob

from tools.io_tools import mkdir_if_missing
from tools.net_tools import remove_prefix, check_keys
from config.parameters import OUTPUT_DIR
from tools.video_tools import load_video_bgr, load_half_video_bgr

import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
import gc
import joblib
import dlib

#
#
# global config
#
#
MIN_SIZE = 54

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
torch.set_grad_enabled(False)
cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

#
# detect face
#
which_dataset = 'vipl'

if which_dataset == 'ubfc2':
    video_path_list = ubfc2.UBFC2.video_path_list
elif which_dataset == 'pure':
    video_path_list = pure.Pure.video_dir_list
elif which_dataset == 'vipl':
    video_path_list = vipl.Vipl.video_path_list
# elif which_dataset == 'mahnob':
#     video_path_list = mahnob.Mahnob.video_path_list
else:
    raise Exception

output_dir = os.path.join(
    OUTPUT_DIR,
    'preprocess_detect',
    which_dataset,
)


#
#
#


def load_model():
    device = "cuda"
    pretrained_path = '../attachment/pytorch_face_landmark/Retinaface/weights/mobilenet0.25_Final.pth'
    model = RetinaFace(cfg=cfg_mnet, phase='test')
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    model.eval()
    model = model.to(device)
    return model


def get_largest_face(faces, threshold=0.9):
    if len(faces) == 0:
        return []
    elif len(faces) == 1:
        return faces[0]
    else:
        largest_face_idx = -1
        largest_face_area = -1
        for k, face in enumerate(faces):
            if face[4] < threshold:  # remove low confidence detection
                continue
            x1 = int(face[0])
            y1 = int(face[1])
            x2 = int(face[2])
            y2 = int(face[3])
            rect = dlib.rectangle(x1, y1, x2, y2)
            now_area = rect.area()
            if now_area > largest_face_area:
                largest_face_area = now_area
                largest_face_idx = k
        return faces[largest_face_idx]


class DetectDataset(Dataset):
    def __init__(self, video):
        super(DetectDataset, self).__init__()
        self.video = video
        self.num_frames = len(self.video)
        self.device = 'cuda'

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        img_ = self.video[index]
        img_ = img_.astype('float32')
        img_ -= (104, 117, 123)
        img_ = img_.transpose(2, 0, 1)
        img_ = torch.from_numpy(img_)
        img_ = img_.to(self.device)
        return img_


def detect_face():
    retina_face_model = load_model()
    device = "cuda"
    resize = 1
    confidence_threshold = 0.05
    top_k = 5000
    keep_top_k = 750
    nms_threshold = 0.3
    vis_threshold = 0.5

    # detect
    for v_idx, v_path in enumerate(video_path_list):
        if v_idx not in [237,
                         248,
                         348,
                         503,
                         1114, ]:
            continue
        print('Process: v_idx = {}, video_path = "{}"'.format(v_idx, v_path))
        output_video_dir = os.path.join(
            output_dir,
            str(v_idx)
        )
        mkdir_if_missing(output_video_dir)
        print('output_video_dir = "{}"'.format(output_video_dir))

        if which_dataset in ['pure']:
            video = pure.load_video_bgr(v_path)
        elif which_dataset in ['mahnob']:
            video = load_half_video_bgr(v_path)
        else:
            video = load_video_bgr(v_path)

        dataset = DetectDataset(video=video)
        data_loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            drop_last=False,
        )

        img = video[0]
        im_height, im_width, _ = img.shape
        scale = torch.tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], dtype=torch.float32,
                             device=device)
        prior_box = PriorBox(cfg_mnet, image_size=(im_height, im_width))
        priors = prior_box.forward()
        priors = priors.to(device)
        prior_data = priors.data

        start_idx = 0
        for idx, batch in enumerate(data_loader):
            loc_b, conf_b, land_ms_b = retina_face_model(batch)
            for jdx in range(len(loc_b)):
                loc = loc_b[jdx]
                conf = conf_b[jdx]
                boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
                boxes = boxes * scale / resize
                boxes = boxes.cpu().numpy()
                scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

                # ignore low scores
                inds = np.where(scores > confidence_threshold)[0]
                boxes = boxes[inds]
                scores = scores[inds]

                # keep top-K before NMS
                order = scores.argsort()[::-1][:top_k]
                boxes = boxes[order]
                scores = scores[order]

                # do NMS
                dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                keep = py_cpu_nms(dets, nms_threshold)
                dets = dets[keep, :]

                # keep top-K faster NMS
                dets = dets[:keep_top_k, :]

                # filter using vis_threshold
                det_bboxes = []
                for b in dets:
                    if b[4] > vis_threshold:
                        x_min, y_min, x_max, y_max, score = b[0], b[1], b[2], b[3], b[4]
                        bbox = [x_min, y_min, x_max, y_max, score]
                        det_bboxes.append(bbox)

                if len(det_bboxes) == 0:
                    print('Detect nothing: box = 0 : v_idx = {}, start_idx = {}'.format(v_idx, start_idx))
                    start_idx += 1
                    continue
                else:
                    face = get_largest_face(det_bboxes, threshold=0.9)
                    x1 = int(face[0])
                    y1 = int(face[1])
                    x2 = int(face[2])
                    y2 = int(face[3])
                    rect = dlib.rectangle(x1, y1, x2, y2)

                    if rect.width() < MIN_SIZE or rect.height() < MIN_SIZE:
                        print('Detect nothing: too small : v_idx = {}, start_idx = {}'.format(v_idx, start_idx))
                        start_idx += 1
                        continue
                    else:
                        output_video_frame_path = os.path.join(
                            output_video_dir,
                            str(start_idx)
                        )
                        joblib.dump(face, output_video_frame_path)
                        start_idx += 1
                        continue

        del video
        gc.collect()


if __name__ == '__main__':
    detect_face()
