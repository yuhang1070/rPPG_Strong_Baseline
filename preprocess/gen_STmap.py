from tools.face_tools import merge_add_mask
import copy
import cv2
import numpy as np
from pyDOE2 import fullfact
import matplotlib.pyplot as plt

from tools.skin_tools import detect_skin_bob


def get_ROI_signal(img, mask):
    m, n, c = img.shape
    mask = mask.reshape([m, n]).astype('uint8')

    signal = np.zeros((1, c))
    for i in range(c):
        tmp = img[:, :, i]
        signal[0, i] = np.mean(tmp[np.where(mask)])
    return signal


def poly2mask(_lmk, _shape, val=1, b_val=0):
    if b_val == 0:
        hull_mask = np.zeros(_shape[0:2] + (1,), dtype=np.float32)
    else:
        hull_mask = np.ones(_shape[0:2] + (1,), dtype=np.float32)
    cv2.fillPoly(hull_mask, [_lmk], (val,))
    return hull_mask


# def moving_average_3(x, ):
#     res = np.convolve(x, np.ones(3), 'same') / 3
#     res[0] = np.mean(x[:2])
#     res[-1] = np.mean(x[-2:])
#     return res


def get_combined_signal_map(SignalMap, ROI_num):
    All_idx = fullfact([2, ] * ROI_num.shape[0])
    for i in range(All_idx.shape[0]):
        All_idx[i, :] = All_idx[i, ::-1]

    SignalMapOut = np.zeros([All_idx.shape[0] - 1, 1, SignalMap.shape[1]])

    for i in range(1, All_idx.shape[0]):
        tmp_idx = np.where(All_idx[i, :] == 1)
        tmp_signal = SignalMap[tmp_idx]
        tmp_ROI = ROI_num[tmp_idx]
        tmp_ROI = tmp_ROI / np.sum(tmp_ROI)
        SignalMapOut[i - 1, :, :] = np.sum(tmp_signal * tmp_ROI, axis=0)
    return SignalMapOut


# def get_yuv(r_channel, g_channel, b_channel):
#     # Conversion Formula
#     y_channel = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
#     u_channel = 128 - 0.168736 * r_channel - 0.331264 * g_channel + 0.5 * b_channel
#     v_channel = 128 + 0.5 * r_channel - 0.418688 * g_channel - 0.081312 * b_channel
#     return y_channel, u_channel, v_channel


# def get_YCbCr(r_channel, g_channel, b_channel):
#     # Conversion Formula
#     Y_channel = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
#     Cb_channel = 0.568 * (b_channel - Y_channel) + 128
#     Cr_channel = 0.713 * (r_channel - Y_channel) + 128
#     return Y_channel, Cb_channel, Cr_channel


def cal_line_1d(x1, x2, n=2):
    res = [x1]
    length = (x2 - x1) / n
    for i in range(n - 1):
        res.append(
            x1 + length * (i + 1)
        )
    res.append(x2)
    return res


def cal_line_2d(p1, p2, n=2):
    res0 = cal_line_1d(p1[0], p2[0], n=n)
    res1 = cal_line_1d(p1[1], p2[1], n=n)
    return list(zip(res0, res1))


def cal_landmarks(lh, rh, ll, rl, n=2, n_y=1):
    res_x0 = cal_line_2d(lh, rh, n=n)
    res_x1 = cal_line_2d(ll, rl, n=n)
    lmks = []
    for i in range(n):
        res_y0 = cal_line_2d(res_x0[i], res_x1[i], n=n_y)
        res_y1 = cal_line_2d(res_x0[i + 1], res_x1[i + 1], n=n_y)
        for j in range(n_y):
            y_lmk = np.array([res_y0[j], res_y0[j + 1], res_y1[j + 1], res_y1[j]], dtype=np.int32)
            lmks.append(y_lmk)
    return lmks


def cal_all_landmarks(ipt_list):
    lmks = []
    for lh, rh, ll, rl, n, n_y in ipt_list:
        tmp_lmks = cal_landmarks(
            lh=lh,
            rh=rh,
            ll=ll,
            rl=rl,
            n=n,
            n_y=n_y,
        )
        for lmk in tmp_lmks:
            lmks.append(lmk)
    return lmks


def cal_pixel_num_and_mean(image, landmarks, skin_mask=None):
    # crop
    (x, y, w, h) = cv2.boundingRect(landmarks)
    image = image[y: y + h, x:x + w]
    landmarks = [[landmarks[_][0] - x, landmarks[_][1] - y] for _ in range(len(landmarks))]
    # mask
    landmarks = np.array(landmarks, dtype=np.int32)
    mask = poly2mask(landmarks, image.shape)

    if skin_mask is not None:
        # crop
        tmp_skin_mask = skin_mask[y: y + h, x:x + w]
        # merge
        mask = np.logical_and.reduce([
            tmp_skin_mask.reshape(tmp_skin_mask.shape + (1,)),
            mask,
        ])
    # pixel num
    pixel_num = np.sum(mask)
    # yuv
    image = image.astype('float32')
    # pixel mean
    pixel_mean_bgr = get_ROI_signal(image, mask).reshape([-1])
    # print(pixel_mean_vuy)
    # r_mean = pixel_mean_bgr[2]
    # g_mean = pixel_mean_bgr[1]
    # b_mean = pixel_mean_bgr[0]
    #
    # y_mean, u_mean, v_mean = get_yuv(
    #     r_channel=r_mean,
    #     g_channel=g_mean,
    #     b_channel=b_mean,
    # )
    #
    # pixel_mean_vuy = np.array(
    #     [
    #         v_mean,
    #         u_mean,
    #         y_mean,
    #     ],
    # )
    # 187 [146.88026344 118.99295405 145.52420562]
    return pixel_num, pixel_mean_bgr


def image_to_STmap_68(
        image,
        landmarks,
        roi_x,
        roi_y,
        flag_plot=False,
        flag_segment=False,
        skin_threshold=0,
):
    lmk_idx_forehead = list(range(17, 26 + 1))
    lmk_idx_left_eye = list(range(36, 41 + 1))
    lmk_idx_right_eye = list(range(42, 47 + 1))

    lmk_left_eye = landmarks[lmk_idx_left_eye]
    lmk_right_eye = landmarks[lmk_idx_right_eye]
    lmk_forehead = landmarks[lmk_idx_forehead]
    lmk_left_eye_mean = np.mean(lmk_left_eye, axis=0)
    lmk_right_eye_mean = np.mean(lmk_right_eye, axis=0)
    eye_distance = np.linalg.norm(lmk_left_eye_mean - lmk_right_eye_mean, ord=2)
    tmp = (np.mean(landmarks[list(range(17, 21 + 1))], axis=0) +
           np.mean(landmarks[list(range(22, 26 + 1))], axis=0)) / 2 - \
          (lmk_left_eye_mean + lmk_right_eye_mean) / 2
    tmp = eye_distance / np.linalg.norm(tmp) * 0.6 * tmp
    lmk_forehead = np.array(np.concatenate(
        [
            lmk_forehead.reshape([-1, 2]),
            (lmk_forehead[-1] + tmp).reshape([-1, 2]),
            (lmk_forehead[0] + tmp).reshape([-1, 2]),
            lmk_forehead[0].reshape([-1, 2]),
        ]
    ), dtype=np.int32).reshape([-1, 2])
    lmk_forehead[lmk_forehead < 0] = 0
    #
    # 0-crop
    #
    (x, y, w, h) = cv2.boundingRect(
        np.concatenate(
            [landmarks,
             lmk_forehead, ],
            axis=0
        )
    )
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    image = image[y: y + h, x:x + w]
    landmarks = [[landmarks[_][0] - x, landmarks[_][1] - y] for _ in range(len(landmarks))]
    landmarks = np.array(landmarks, dtype=np.int32)

    lmk_forehead = [[lmk_forehead[_][0] - x, lmk_forehead[_][1] - y] for _ in range(len(lmk_forehead))]
    lmk_forehead = np.array(lmk_forehead, dtype=np.int32)

    #
    # 1-roi
    #
    lmks = cal_all_landmarks(
        [
            #
            # forehead
            #
            [lmk_forehead[11], lmk_forehead[0], lmk_forehead[10], lmk_forehead[9], 2 * roi_x, 5 * roi_y],

            # t0
            [landmarks[1], landmarks[28], landmarks[2], landmarks[29], 3 * roi_x, roi_y],
            [landmarks[28], landmarks[15], landmarks[29], landmarks[14], 3 * roi_x, roi_y],
            # t1
            [landmarks[2], landmarks[29], landmarks[3], landmarks[30], 3 * roi_x, roi_y],
            [landmarks[29], landmarks[14], landmarks[30], landmarks[13], 3 * roi_x, roi_y],
            # # t2
            [landmarks[3], landmarks[30], landmarks[4], landmarks[51], 3 * roi_x, roi_y],
            [landmarks[30], landmarks[13], landmarks[51], landmarks[12], 3 * roi_x, roi_y],

            #
            [landmarks[4],
             [(landmarks[4][0] + landmarks[51][0]) // 2, (landmarks[4][1] + landmarks[51][1]) // 2],
             landmarks[6], landmarks[59], 1 * roi_x, roi_y],
            [landmarks[12],
             [(landmarks[12][0] + landmarks[51][0]) // 2, (landmarks[12][1] + landmarks[51][1]) // 2],
             landmarks[10], landmarks[55], 1 * roi_x, roi_y],

            #
            [landmarks[6], landmarks[59], landmarks[8], landmarks[57], 1 * roi_x, roi_y],
            [landmarks[10], landmarks[55], landmarks[8], landmarks[57], 1 * roi_x, roi_y],

        ]
    )

    if flag_plot:
        image_copy = copy.deepcopy(image)
        thickness = 1
        lmk_mean_list = []
        for lmk_idx, lmk in enumerate(lmks):
            cv2.polylines(image_copy, pts=[lmk], isClosed=True, thickness=thickness, color=(0, 0, 255))
            lmk_mean = np.mean(lmk, axis=0).astype('int32')
            # print(lmk_mean)
            lmk_mean_list.append(lmk_mean)
            # cv2.putText(image_copy, ''.format(lmk_idx), lmk_mean, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # plt.text(lmk_mean[0], lmk_mean[1], )
        plt.xlabel('An example of ROI visualisation.')
        plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        # for lmk_idx, lmk in enumerate(lmk_mean_list):
        #     plt.text(lmk[0], lmk[1], '{}'.format(lmk_idx))
        plt.show()

    #
    # 2-roi-crop
    #
    forehead_mask = poly2mask(lmks[0], image.shape)
    for lmk in lmks[1:10]:
        tmp_mask = poly2mask(lmk, image.shape)
        forehead_mask = np.logical_or.reduce(
            [
                forehead_mask,
                tmp_mask,
            ]
        )
    forehead_mask1 = poly2mask(lmk_forehead, image.shape, val=1, b_val=0)
    forehead_mask = np.logical_and.reduce(
        [
            forehead_mask,
            forehead_mask1,
        ]
    )
    roi_mask = poly2mask(lmks[10], image.shape)
    for lmk in lmks[10 + 1:]:
        tmp_mask = poly2mask(lmk, image.shape)
        roi_mask = np.logical_or.reduce(
            [
                roi_mask,
                tmp_mask,
            ]
        )
    roi_mask = np.logical_or.reduce(
        [
            forehead_mask,
            roi_mask
        ]
    )
    mouse_landmarks = landmarks[[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 48, ]]
    # print(mouse_landmarks)
    mouse_mask = poly2mask(mouse_landmarks, image.shape, val=0, b_val=1)
    roi_mask = np.logical_and.reduce(
        [
            roi_mask,
            mouse_mask,
        ]
    )

    # plt.imshow(merge_add_mask(image, mouse_mask))
    # plt.show()

    image = merge_add_mask(image, roi_mask)
    if flag_plot:
        image_copy = copy.deepcopy(image)
        thickness = 1
        for lmk in lmks:
            cv2.polylines(image_copy, pts=[lmk], isClosed=True, thickness=thickness, color=(0, 0, 255))
        plt.xlabel('2-roi-crop')
        plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        plt.show()

    #
    # 3-skin-crop
    #
    if flag_segment:
        skin_mask = detect_skin_bob(image, skin_threshold)
        image = merge_add_mask(image, skin_mask)
        # if flag_plot:
        #     image_copy = copy.deepcopy(image)
        #     plt.xlabel('3-skin-crop')
        #     plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        #     plt.show()
        if flag_plot:
            image_copy = copy.deepcopy(image)
            thickness = 1
            for lmk in lmks:
                cv2.polylines(image_copy, pts=[lmk], isClosed=True, thickness=thickness, color=(0, 0, 255))
            plt.xlabel('3-skin-crop')
            plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
            plt.show()
    else:
        skin_mask = None
    if flag_plot:
        exit(0)

    #
    # 4-cal-mean
    #
    # pixel_mean_vuy_list = []
    pixel_mean_bgr_list = []
    for i, lmk in enumerate(lmks):
        # 187 [146.88026344 118.99295405 145.52420562]
        pixel_num, pixel_mean_bgr = cal_pixel_num_and_mean(image, lmk, skin_mask=skin_mask)
        # print(i, pixel_num, pixel_mean)
        # pixel_mean_vuy_list.append(pixel_mean_vuy)
        pixel_mean_bgr_list.append(pixel_mean_bgr)

    # pixel_mean_vuy_list = np.array(pixel_mean_vuy_list, dtype=np.float32)  # (40, 3) yuv
    pixel_mean_bgr_list = np.array(pixel_mean_bgr_list, dtype=np.float32)
    # print(pixel_mean_list.shape)

    return pixel_mean_bgr_list
