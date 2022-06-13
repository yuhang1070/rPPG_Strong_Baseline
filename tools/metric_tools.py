from scipy import interpolate
import numpy as np
from scipy.signal import find_peaks


def get_ibi(peak, sig_fps):
    peak_loc = np.where(peak == 1)[0]
    IPI = (peak_loc[1:] - peak_loc[0:-1]) / sig_fps
    T = np.zeros(len(IPI))
    for i in range(len(IPI)):
        T[i] = np.sum(IPI[:i])

    IBI = np.array([T, IPI])  # x-axis is time, y-axis is ibi
    return IBI


def ibi_error(ibi, ibi_gt):
    f = interpolate.interp1d(ibi[0], 1000 * ibi[1])  # convert to ms
    f_gt = interpolate.interp1d(ibi_gt[0], 1000 * ibi_gt[1])

    t_min = np.max([ibi[0].min(), ibi_gt[0].min()])
    t_max = np.min([ibi[0].max(), ibi_gt[0].max()])  # find the interpolation x-axis range

    inter_ibi = f(np.linspace(t_min, t_max, 100))
    inter_ibi_gt = f_gt(np.linspace(t_min, t_max, 100))  # interpolation

    ibi_err = np.mean(np.abs(inter_ibi - inter_ibi_gt))  # compare absolute error between the 2 interpolated IBI curves
    return ibi_err


def hr_error(ibi, ibi_gt):
    hr = 60 / np.mean(ibi[1])
    hr_gt = 60 / np.mean(ibi_gt[1])  # compute heart rate from ibi
    hr_err = np.abs(hr - hr_gt)
    return hr, hr_gt, hr_err


def compute_metric_from_bvp(pred_bvp, gt_bvp, fps):
    ibi_err_list = []
    hr_err_list = []
    hr_list = []
    hr_gt_list = []

    pred_bvp_len = pred_bvp.shape[0]

    for p_idx in range(pred_bvp_len):
        pred_bvp_i = pred_bvp[p_idx]
        gt_bvp_i = gt_bvp[p_idx]
        fps_i = fps[p_idx]

        pred_bvp_i = pred_bvp_i.reshape([-1])
        gt_bvp_i = gt_bvp_i.reshape([-1])
        fps_i = fps_i.reshape([-1])

        # pred_bvp_i = np.cumsum(pred_bvp_i)
        # gt_bvp_i = np.cumsum(gt_bvp_i)

        # import matplotlib.pyplot as plt
        #
        # plt.plot(pred_bvp_i, 'r.-')
        # plt.plot(gt_bvp_i, 'g*-')
        # plt.show()
        # plt.close()
        # pred_bvp_i = np.float64(pred_bvp_i)
        # pred_bvp_i = detrend(sig=pred_bvp_i, Lambda=100)

        distance_i = fps_i / 2
        height_gt_i = np.mean(gt_bvp_i)
        height_pred_i = np.mean(pred_bvp_i)

        peaks_pred_i = find_peaks(pred_bvp_i, height=height_pred_i, distance=distance_i)[0]
        peaks_gt_i = find_peaks(gt_bvp_i, height=height_gt_i, distance=distance_i)[0]

        if isinstance(peaks_pred_i, int) or peaks_pred_i.shape[0] <= 2:
            continue

        ibi_pred_i = np.zeros([len(pred_bvp_i)], dtype=np.float32)
        ibi_gt_i = np.zeros([len(gt_bvp_i)], dtype=np.float32)

        ibi_pred_i[peaks_pred_i] = 1.0
        ibi_gt_i[peaks_gt_i] = 1.0

        ibi_pred_ii = get_ibi(ibi_pred_i, sig_fps=fps_i)
        ibi_gt_ii = get_ibi(ibi_gt_i, sig_fps=fps_i)

        #### compute IBI error and HR error ####
        ibi_err = ibi_error(ibi_pred_ii, ibi_gt_ii)
        hr, hr_gt, hr_err = hr_error(ibi_pred_ii, ibi_gt_ii)

        ibi_err_list.append(ibi_err)
        hr_err_list.append(hr_err)
        hr_list.append(hr)
        hr_gt_list.append(hr_gt)

    return ibi_err_list, hr_err_list, hr_list, hr_gt_list


if __name__ == '__main__':
    pred_bvp = np.zeros([32, 256])  # ()
    bvp = np.zeros([32, 256])  # (32, 256)
    fps = np.ones([32]) * 30  # (32,)
    print(compute_metric_from_bvp(pred_bvp, bvp, fps))
