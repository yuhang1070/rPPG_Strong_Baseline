import cv2 as cv
import numpy as np


def trans_format_gray(image, image_format):
    assert image_format in ['bgr', 'rgb', 'gray']
    if image_format == 'bgr':
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    elif image_format == 'rgb':
        image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    elif image_format == 'gray':
        image_gray = image
    else:
        raise Exception
    return image_gray


def trans_format_bgr(image, image_format):
    assert image_format in ['bgr', 'rgb']
    if image_format == 'bgr':
        image_bgr = image
    elif image_format == 'rgb':
        image_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    else:
        raise Exception
    return image_bgr


def trans_format_rgb(image, image_format):
    assert image_format in ['bgr', 'rgb']
    if image_format == 'bgr':
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    elif image_format == 'rgb':
        image_rgb = image
    else:
        raise Exception
    return image_rgb
