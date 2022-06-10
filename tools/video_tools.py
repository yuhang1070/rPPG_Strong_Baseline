import cv2 as cv
import numpy as np
import os


#
# common
#

def check_file_exist(path: str):
    if not os.path.exists(path):
        raise Exception('Can not find path: "{}"'.format(path))


def trans_fourcc(fourcc: str) -> str:
    fourcc = int(fourcc)
    fourcc = chr((fourcc & 0XFF)) + chr((fourcc & 0XFF00) >> 8) + chr((fourcc & 0XFF0000) >> 16) + chr(
        (fourcc & 0XFF000000) >> 24) + chr(0)
    fourcc = fourcc[:4]
    return fourcc


def frames_ps_video(path: str) -> int:
    check_file_exist(path)
    video = cv.VideoCapture(path)
    fps = video.get(cv.CAP_PROP_FPS)
    # fps = int(round(fps))
    video.release()
    return fps


def metadata_video(path: str) -> dict:
    check_file_exist(path)
    metadata = {}
    video = cv.VideoCapture(path)
    # num_frames
    num_frames = video.get(cv.CAP_PROP_FRAME_COUNT)
    num_frames = int(num_frames)
    metadata['num_frames'] = num_frames
    # height
    height = video.get(cv.CAP_PROP_FRAME_HEIGHT)
    height = int(height)
    metadata['height'] = height
    # width
    width = video.get(cv.CAP_PROP_FRAME_WIDTH)
    width = int(width)
    metadata['width'] = width
    # frame_rate
    frame_rate = video.get(cv.CAP_PROP_FPS)
    # frame_rate = int(round(frame_rate))
    metadata['frame_rate'] = frame_rate
    # duration
    duration = num_frames / frame_rate
    metadata['duration'] = duration
    # fourcc
    fourcc = video.get(cv.CAP_PROP_FOURCC)
    fourcc = trans_fourcc(fourcc)
    metadata['fourcc'] = fourcc
    # release video
    video.release()
    return metadata


def load_video_bgr(path: str) -> list:
    check_file_exist(path)
    video = cv.VideoCapture(path)
    frames = []
    ret_val, frame = video.read()
    while ret_val:
        frames.append(frame)
        ret_val, frame = video.read()
    video.release()
    return frames


def load_half_video_bgr(path: str) -> list:
    check_file_exist(path)
    video = cv.VideoCapture(path)
    frames = []
    ret_val, frame = video.read()
    idx = 0
    while ret_val:
        if idx % 2 == 0:
            frames.append(frame)
        ret_val, frame = video.read()
        idx += 1
    video.release()
    return frames


def load_video_idx_bgr(path: str, index: int) -> np.ndarray:
    check_file_exist(path)
    video = cv.VideoCapture(path)
    video.set(cv.CAP_PROP_POS_FRAMES, index)
    ret_val, frame = video.read()
    video.release()
    return frame


def num_frames_video(path: str) -> int:
    # check_file_exist(path)
    # num_frames = 0
    # video = cv.VideoCapture(path)
    # ret_val, frame = video.read()
    # while ret_val:
    #     num_frames += 1
    #     ret_val, frame = video.read()
    # video.release()
    # return num_frames
    check_file_exist(path)
    video = cv.VideoCapture(path)
    # num_frames
    num_frames = video.get(cv.CAP_PROP_FRAME_COUNT)
    video.release()
    num_frames = int(num_frames)
    return num_frames
