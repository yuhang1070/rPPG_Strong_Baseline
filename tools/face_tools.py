import numpy as np
import cv2 as cv


def get_image_hull_mask(image_shape, image_landmarks, ):
    if image_landmarks.shape[0] != 68:
        raise Exception('get_image_hull_mask works only with 68 landmarks')
    hull_mask = np.zeros(image_shape[0:2] + (1,), dtype=np.float32)

    lmrks = image_landmarks.astype('int32')

    r_jaw = (lmrks[0:9], lmrks[17:18])
    l_jaw = (lmrks[8:17], lmrks[26:27])
    r_cheek = (lmrks[17:20], lmrks[8:9])
    l_cheek = (lmrks[24:27], lmrks[8:9])
    nose_ridge = (lmrks[19:25], lmrks[8:9],)
    r_eye = (lmrks[17:22], lmrks[27:28], lmrks[31:36], lmrks[8:9])
    l_eye = (lmrks[22:27], lmrks[27:28], lmrks[31:36], lmrks[8:9])
    nose = (lmrks[27:31], lmrks[31:36])

    parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]

    for item in parts:
        merged = np.concatenate(item)
        cv.fillConvexPoly(hull_mask, cv.convexHull(merged), (1,))

    return hull_mask


def merge_add_mask(img_1, mask):
    assert mask is not None
    mask = mask.astype('uint8')
    mask = mask * 255
    b_channel, g_channel, r_channel = cv.split(img_1)
    r_channel = cv.bitwise_and(r_channel, r_channel, mask=mask)
    g_channel = cv.bitwise_and(g_channel, g_channel, mask=mask)
    b_channel = cv.bitwise_and(b_channel, b_channel, mask=mask)
    res_img = cv.merge((b_channel, g_channel, r_channel))
    return res_img


def get_part(image_, part_):
    (x, y, w, h) = cv.boundingRect(part_)
    image = image_[y: y + h, x:x + w]
    image_shape = image.shape

    part_ = [[part_[_][0] - x, part_[_][1] - y] for _ in range(len(part_))]
    part_ = np.array(part_, dtype=np.int32)

    hull_mask = np.zeros(image_shape[0:2] + (1,), dtype=np.float32)

    cv.fillConvexPoly(hull_mask, cv.convexHull(part_), (1,))

    image = merge_add_mask(image, hull_mask)

    return image


def get_left_cheek1(image, image_landmarks, ):
    lmrks = image_landmarks.astype('int32')
    l_cheek1 = (
        lmrks[0:2 + 1],
        lmrks[31:31 + 1],
        lmrks[41:41 + 1],
    )
    part = np.concatenate(l_cheek1)

    return get_part(image, part)


def get_left_cheek2(image, image_landmarks, ):
    lmrks = image_landmarks.astype('int32')
    l_cheek2 = (
        lmrks[2:5 + 1],
        lmrks[48:48 + 1],
        lmrks[31:31 + 1],
    )
    part = np.concatenate(l_cheek2)

    return get_part(image, part)


def get_right_cheek1(image, image_landmarks, ):
    lmrks = image_landmarks.astype('int32')
    r_cheek1 = (
        lmrks[46:46 + 1],
        lmrks[35:35 + 1],
        lmrks[14:16 + 1],
    )
    part = np.concatenate(r_cheek1)

    return get_part(image, part)


def get_right_cheek2(image, image_landmarks, ):
    lmrks = image_landmarks.astype('int32')
    r_cheek2 = (
        lmrks[35:35 + 1],
        lmrks[54:54 + 1],
        lmrks[11:14 + 1],
    )
    part = np.concatenate(r_cheek2)

    return get_part(image, part)


def get_jaw(image, image_landmarks, ):
    lmrks = image_landmarks.astype('int32')
    jaw = (
        lmrks[5:11 + 1],
        lmrks[54:59 + 1],
        lmrks[48:48 + 1],
        lmrks[5:5 + 1],
    )
    part = np.concatenate(jaw)

    return get_part(image, part)
