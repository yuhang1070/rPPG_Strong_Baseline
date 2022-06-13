from scipy import signal
import numpy as np
from scipy.sparse import spdiags
import numba as nb


def butter_bandpass_filter(data, low_cut, high_cut, fs, order=5):
    """
    :param data: signal
    :param low_cut: low cut of filter
    :param high_cut: high cut of filter
    """
    if data is None:
        raise TypeError("Please specify an input signal.")

    # float
    fs = float(fs)

    # filter
    f_nyq = 0.5 * fs
    f_low = low_cut / f_nyq
    f_high = high_cut / f_nyq
    [b, a] = signal.butter(N=order, Wn=[f_low, f_high], btype='bandpass')

    # filter
    filter_data = signal.filtfilt(b, a, data, method="pad")
    return filter_data


def get_peaks(data, fs=30):
    """
    REFER:
    https://github.com/phuselab/pyVHR/blob/master/pyVHR/signals/bvp.py
    :param data: data
    :param fs: fs
    :return: peaks
    """
    distance = fs / 2
    height = np.mean(data)
    peaks, _ = signal.find_peaks(data, height=height, distance=distance)
    return peaks


@nb.njit()
def get_spdiags(signal_len, ):
    arr = np.zeros((signal_len - 2, signal_len), dtype=np.float32)
    for i in range(signal_len - 2):
        arr[i, i:i + 3] = np.array((1., -2., 1.), dtype=np.float32)
    return arr


def get_detrend_param(signal_length=256, Lambda=100):
    # signal_length = 256

    # observation matrix
    H = np.identity(signal_length, dtype=np.float32)

    D = get_spdiags(signal_length)
    default_param = (H - np.linalg.inv(H + (Lambda * Lambda) * np.dot(D.T, D)))
    return default_param


de_param = get_detrend_param(signal_length=256, Lambda=100)


# @nb.njit()
def detrend(sig, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This code is based on the following article
    "An advanced detrending method with application to HRV analysis". Tarvainen et al.,
    IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = sig.shape[0]
    #
    # # observation matrix
    # H = np.identity(signal_length)
    #
    # D = get_spdiags(signal_length)
    # default_param = (H - np.linalg.inv(H + (Lambda * Lambda) * np.dot(D.T, D)))
    filtered_signal = np.dot(de_param, sig)
    return filtered_signal


def demo():
    sig = np.zeros([256])
    sig = detrend(sig, 1000)

    print(de_param.shape)


if __name__ == '__main__':
    demo()
