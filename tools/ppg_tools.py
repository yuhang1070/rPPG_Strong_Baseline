from scipy.interpolate import Akima1DInterpolator
from tools.metric_tools import get_ibi
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import periodogram


def base_bvp_hr(bvp, fps):
    low_bound = 40
    high_bound = 150
    time_length = len(bvp)
    # clip_length = 300
    # delta = 3
    pi = 3.14159265

    bpm_range = np.arange(low_bound, high_bound, dtype=np.float32, ) / 60.0
    two_pi_n = (2 * pi) * np.arange(0, time_length, dtype=np.float32, )
    hanning = np.hanning(time_length)

    f_t = bpm_range / fps
    f_t = f_t.reshape([-1, 1])
    preds = bvp * hanning
    # preds = bvp
    tmp = two_pi_n
    tmp = tmp.reshape([1, -1])

    complex_absolute = np.sum(preds * np.sin(f_t * tmp), axis=-1) ** 2 \
                       + np.sum(preds * np.cos(f_t * tmp), axis=-1) ** 2

    whole_max_idx = complex_absolute.argmax()
    whole_max_idx = whole_max_idx + low_bound

    return whole_max_idx


#
# Code from: REPSS-2021
#
def ibi_compute_hr(bvp, fps):
    """
    Code from: https://github.com/sunhm15/REPSS-2021
    :param bvp:
    :param fps:
    :return: hr
    """
    distance = fps / 2
    height = np.mean(bvp)
    peaks = find_peaks(bvp, height=height, distance=distance)[0]

    ibi = np.zeros([len(bvp)], dtype=np.float32)
    ibi[peaks] = 1.0

    ibi = get_ibi(ibi, sig_fps=fps)
    hr = 60 / np.mean(ibi[1])
    return hr


#
# Code from: iphys-tools
#
def psd_compute_hr(BVP, FS, LL_PR=40.0, UL_PR=150.0):
    """
    code from: https://github.com/danmcduff/iphys-toolbox
    :param BVP: A BVP timeseries.
    :param FS: The sample rate of the BVP time series (Hz/fps).
    :param LL_PR: The lower limit for pulse rate (bpm).
    :param UL_PR: The upper limit for pulse rate (bpm).
    :return:
    PR                      = The estimated PR in BPM.
    """
    assert len(BVP.shape) == 1

    Nyquist = FS / 2
    FResBPM = 0.01  # resolution (bpm) of bins in power spectrum used to determine PR and SNR

    N = (60 * 2 * Nyquist) / FResBPM

    # Construct Periodogram
    F, Pxx = periodogram(x=BVP, window='hann', nfft=N, fs=FS, return_onesided=False, detrend=False)
    # print('N: {}, X: {}'.format(N, BVP.shape[-1]))
    FMask = (F >= (LL_PR / 60)) & (F <= (UL_PR / 60))

    # Calculate predicted HR:
    FRange = F[FMask]
    # PRange = Pxx[FMask]
    MaxInd = np.argmax(Pxx[FMask], 0)
    PR_F = FRange[MaxInd]
    PR = PR_F * 60
    return PR


#
# Code from: RemotePPG
#
def compute_power_spectrum(signal, Fs, zero_pad=None):
    if zero_pad is not None:
        L = len(signal)
        signal = np.pad(signal, (int(zero_pad / 2 * L), int(zero_pad / 2 * L)), 'constant')
    freqs = np.fft.fftfreq(len(signal), 1 / Fs) * 60  # in bpm
    ps = np.abs(np.fft.fft(signal)) ** 2
    cutoff = len(freqs) // 2
    freqs = freqs[:cutoff]
    ps = ps[:cutoff]
    return freqs, ps


def RemotePPG_compute_hr(signal, Fs, min_hr=40., max_hr=180., method='fast_ideal'):
    if method == 'ideal':
        """ Zero-pad in time domain for ideal interp in freq domain
        """
        signal = signal - np.mean(signal)
        freqs, ps = compute_power_spectrum(signal, Fs, zero_pad=100)
        cs = Akima1DInterpolator(freqs, ps)
        max_val = -np.Inf
        interval = 0.1
        min_bound = max(min(freqs), min_hr)
        max_bound = min(max(freqs), max_hr) + interval
        for bpm in np.arange(min_bound, max_bound, interval):
            cur_val = cs(bpm)
            if cur_val > max_val:
                max_val = cur_val
                max_bpm = bpm
        return max_bpm

    elif method == 'fast_ideal':
        """ Zero-pad in time domain for ideal interp in freq domain
        """
        signal = signal - np.mean(signal)
        freqs, ps = compute_power_spectrum(signal, Fs, zero_pad=100)
        freqs_valid = np.logical_and(freqs >= min_hr, freqs <= max_hr)
        freqs = freqs[freqs_valid]
        ps = ps[freqs_valid]
        max_ind = np.argmax(ps)
        if 0 < max_ind < len(ps) - 1:
            inds = [-1, 0, 1] + max_ind
            x = ps[inds]
            f = freqs[inds]
            d1 = x[1] - x[0]
            d2 = x[1] - x[2]
            offset = (1 - min(d1, d2) / max(d1, d2)) * (f[1] - f[0])
            if d2 > d1:
                offset *= -1
            max_bpm = f[1] + offset
        elif max_ind == 0:
            x0, x1 = ps[0], ps[1]
            f0, f1 = freqs[0], freqs[1]
            max_bpm = f0 + (x1 / (x0 + x1)) * (f1 - f0)
        elif max_ind == len(ps) - 1:
            x0, x1 = ps[-2], ps[-1]
            f0, f1 = freqs[-2], freqs[-1]
            max_bpm = f0 + (x1 / (x0 + x1)) * (f1 - f0)
        return max_bpm

    elif method == 'fast_ideal_bimodal_filter':
        """ Same as above but check for secondary peak around 1/2 of first
        (to break the tie in case of occasional bimodal PS)
        Note - this may make metrics worse if the power spectrum is relatively flat
        """
        signal = signal - np.mean(signal)
        freqs, ps = compute_power_spectrum(signal, Fs, zero_pad=100)
        freqs_valid = np.logical_and(freqs >= min_hr, freqs <= max_hr)
        freqs = freqs[freqs_valid]
        ps = ps[freqs_valid]
        max_ind = np.argmax(ps)
        max_freq = freqs[max_ind]
        max_ps = ps[max_ind]

        # check for a second lower peak at 0.45-0.55f and >50% power
        freqs_valid = np.logical_and(freqs >= max_freq * 0.45, freqs <= max_freq * 0.55)
        freqs = freqs[freqs_valid]
        ps = ps[freqs_valid]
        if len(freqs) > 0:
            max_ind_lower = np.argmax(ps)
            max_freq_lower = freqs[max_ind_lower]
            max_ps_lower = ps[max_ind_lower]
        else:
            max_ps_lower = 0

        if max_ps_lower / max_ps > 0.50:
            return max_freq_lower
        else:
            return max_freq
    else:
        raise NotImplementedError
