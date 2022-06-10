from config.parameters import MAHNOB_HCI_DIR
from tools.video_tools import load_video_bgr, metadata_video, load_half_video_bgr

import os
import numpy as np
import pyedflib

# database root directory
ROOT_DIR = MAHNOB_HCI_DIR


# Check if video and signal files exist
# Return the video path list and bdf file path list
def check_file(root_dir: str, folder_name_list: list, ):
    video_path_list = []
    bdf_PATH_LIST = []
    for __subject in folder_name_list:
        __subject_dir = os.path.join(root_dir, __subject)
        for __item in os.listdir(__subject_dir):
            if '.avi' in __item:
                video_path_list.append(os.path.join(__subject_dir, __item))
            elif '.bdf' in __item:
                bdf_PATH_LIST.append(os.path.join(__subject_dir, __item))
            else:
                raise Exception('Something error!')
    return video_path_list, bdf_PATH_LIST


# read bdf file
def LoadBDF(bdf_file, name="EXG2", start=None, end=None):
    """
    Reference:
      https://programmerwiki.com/article/60802052016/
      https://www.programmersought.com/article/76255815129/
      https://blog.csdn.net/qq_40995448/article/details/102499561
    """
    with pyedflib.EdfReader(bdf_file) as f:
        status_index = f.getSignalLabels().index('Status')
        sample_frequency = f.samplefrequency(status_index)
        status_size = f.samples_in_file(status_index)
        status = np.zeros((status_size,), dtype='float64')
        f.readsignal(status_index, 0, status_size, status)
        status = status.round().astype('int')
        nz_status = status.nonzero()[0]

        video_start = nz_status[0]
        video_end = nz_status[-1]

        index = f.getSignalLabels().index(name)
        # print('index: {}'.format(index))
        sample_frequency = f.samplefrequency(index)

        video_start_seconds = video_start / sample_frequency

        if start is not None:
            start += video_start_seconds
            start *= sample_frequency
            if start < video_start:
                start = video_start
            start = int(start)
        else:
            start = video_start

        if end is not None:
            end += video_start_seconds
            end *= sample_frequency
            if end > video_end:
                end = video_end
            end = int(end)
        else:
            end = video_end

        PhysicalMax = f.getPhysicalMaximum(index)
        PhysicalMin = f.getPhysicalMinimum(index)
        DigitalMax = f.getDigitalMaximum(index)
        DigitalMin = f.getDigitalMinimum(index)

        scale_factor = (PhysicalMax - PhysicalMin) / (DigitalMax - DigitalMin)
        dc = PhysicalMax - scale_factor * DigitalMax

        container = np.zeros((end - start,), dtype='float')
        f.readsignal(index, start, end - start, container)
        container = container * scale_factor + dc

        return container, sample_frequency


# filter some video frames
def mahnob_select_video(video: list):
    num_frames = len(video)
    new_video = []
    for f_idx in range(num_frames):
        if not (305 <= f_idx < 305 + 61 * 30) or not (f_idx % 2 == 1):
            continue
        new_video.append(video[f_idx])
    return new_video


# load signal
def read_signal_file(filepath: str, name='EXG2'):
    """ Load ECG signal.
        Must return a 1-dim (row array) signal
    """
    # name = 'EXG2'
    # name = 'EXG1'
    ecgTrace, SIG_SampleRate = LoadBDF(filepath, name=name)
    ecgTrace = ecgTrace.astype(np.float32)
    return ecgTrace, SIG_SampleRate


class Mahnob:
    # list of video file names
    folder_name_list = ['2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22', '24', '26', '28', '30', '32',
                        '34', '36', '38', '40', '132', '134', '136', '138', '140', '142', '144', '146', '148', '150',
                        '152', '154', '156', '158', '160', '162', '164', '166', '168', '170', '262', '264', '266',
                        '268', '270', '272', '274', '276', '278', '280', '282', '284', '286', '288', '290', '292',
                        '294', '392', '394', '396', '398', '400', '402', '404', '406', '408', '410', '412', '414',
                        '416', '418', '420', '422', '424', '426', '428', '430', '522', '524', '526', '528', '530',
                        '532', '534', '536', '538', '540', '542', '544', '546', '548', '550', '552', '554', '556',
                        '558', '560', '652', '654', '656', '658', '660', '662', '664', '666', '668', '670', '672',
                        '674', '676', '678', '680', '682', '684', '686', '688', '690', '782', '784', '786', '788',
                        '790', '792', '794', '796', '798', '800', '802', '804', '806', '808', '810', '812', '814',
                        '816', '818', '820', '912', '914', '916', '918', '920', '922', '924', '926', '928', '930',
                        '932', '934', '936', '938', '940', '942', '944', '946', '948', '950', '1042', '1044', '1046',
                        '1048', '1050', '1052', '1054', '1056', '1058', '1060', '1062', '1064', '1066', '1068', '1172',
                        '1174', '1176', '1178', '1180', '1182', '1184', '1186', '1188', '1190', '1192', '1194', '1196',
                        '1198', '1200', '1202', '1204', '1206', '1208', '1210', '1302', '1304', '1306', '1308', '1310',
                        '1312', '1314', '1316', '1318', '1320', '1322', '1324', '1326', '1328', '1330', '1332', '1334',
                        '1336', '1338', '1340', '1562', '1564', '1566', '1568', '1570', '1572', '1574', '1576', '1578',
                        '1580', '1582', '1584', '1586', '1588', '1590', '1592', '1594', '1596', '1598', '1600', '1692',
                        '1694', '1696', '1698', '1700', '1702', '1704', '1706', '1708', '1710', '1712', '1714', '1716',
                        '1718', '1720', '1722', '1724', '1726', '1728', '1730', '1952', '1954', '1956', '1958', '1960',
                        '1962', '1964', '1966', '1968', '1970', '1972', '1974', '1976', '1978', '1980', '1982', '2082',
                        '2084', '2086', '2088', '2090', '2092', '2094', '2096', '2098', '2100', '2102', '2104', '2106',
                        '2108', '2110', '2112', '2114', '2116', '2118', '2120', '2212', '2214', '2216', '2218', '2220',
                        '2222', '2224', '2226', '2228', '2230', '2232', '2234', '2236', '2238', '2240', '2242', '2244',
                        '2246', '2248', '2250', '2342', '2344', '2346', '2348', '2350', '2352', '2354', '2356', '2358',
                        '2360', '2362', '2364', '2366', '2368', '2370', '2372', '2374', '2376', '2378', '2380', '2472',
                        '2474', '2476', '2478', '2480', '2482', '2484', '2486', '2488', '2490', '2492', '2494', '2496',
                        '2498', '2500', '2502', '2504', '2506', '2508', '2510', '2602', '2604', '2606', '2608', '2610',
                        '2612', '2614', '2616', '2618', '2620', '2622', '2624', '2626', '2628', '2630', '2632', '2634',
                        '2636', '2638', '2640', '2732', '2734', '2736', '2738', '2740', '2742', '2744', '2746', '2748',
                        '2750', '2752', '2754', '2756', '2758', '2760', '2762', '2764', '2766', '2768', '2770', '2862',
                        '2864', '2866', '2868', '2870', '2872', '2874', '2876', '2878', '2880', '2882', '2884', '2886',
                        '2888', '2890', '2892', '2894', '2896', '2898', '2900', '2992', '2994', '2996', '2998', '3000',
                        '3002', '3004', '3006', '3008', '3010', '3012', '3014', '3016', '3018', '3020', '3022', '3024',
                        '3026', '3028', '3030', '3122', '3124', '3126', '3128', '3130', '3132', '3134', '3136', '3138',
                        '3140', '3142', '3144', '3146', '3148', '3150', '3152', '3154', '3156', '3158', '3160', '3382',
                        '3384', '3386', '3388', '3390', '3392', '3394', '3396', '3398', '3400', '3402', '3404', '3406',
                        '3408', '3410', '3412', '3414', '3416', '3418', '3420', '3512', '3514', '3516', '3518', '3520',
                        '3522', '3524', '3526', '3528', '3530', '3532', '3534', '3536', '3538', '3540', '3542', '3544',
                        '3546', '3548', '3550', '3642', '3644', '3646', '3648', '3650', '3652', '3654', '3656', '3658',
                        '3660', '3662', '3664', '3666', '3668', '3670', '3672', '3674', '3676', '3678', '3680', '3772',
                        '3774', '3776', '3778', '3780', '3782', '3784', '3786', '3788', '3790', '3792', '3794', '3796',
                        '3798', '3800', '3802', '3804', '3806', '3808', '3810']
    assert len(folder_name_list) == 527

    # Video path list, bdf file path list
    video_path_list, BDF_PATH_LIST = check_file(ROOT_DIR, folder_name_list)
    assert len(video_path_list) == 527
    assert len(BDF_PATH_LIST) == 527

    # load video
    @staticmethod
    def load_video_file(filepath: str, use_part=False) -> list:
        video = load_video_bgr(filepath)
        if use_part:
            video = mahnob_select_video(video)
        return video

    # Read video meta information
    @staticmethod
    def metadata_video(filepath: str) -> dict:
        metadata = metadata_video(filepath)
        return metadata


def demo():
    # print('-' * 50)
    # print('Mahnob')
    # print(Mahnob.folder_name_list)
    # print(len(Mahnob.folder_name_list))
    # print(Mahnob.video_path_list)
    # print(len(Mahnob.video_path_list))
    # print(Mahnob.BDF_PATH_LIST)
    # print(len(Mahnob.BDF_PATH_LIST))
    # print('-' * 50)
    # v_path = Mahnob.video_path_list[0]
    # video = load_half_video_bgr(v_path)
    # print(len(video))
    # video = load_video_bgr(v_path)
    # print(len(video))

    import matplotlib.pyplot as plt

    bvp, gtSigFs = read_signal_file(Mahnob.BDF_PATH_LIST[0])
    plt.plot(bvp[:512])
    plt.show()
    print(bvp.shape, gtSigFs, bvp.dtype)
    print(type(gtSigFs))


"""
MAHNOB-HCI dataset is recorded at 61 fps using a resolution of 780 × 580 and 
includes 527 videos from 27 subjects.

1. To compare fairly with previous works, we follow the same routine in their 
works by using 30 s clip (frames 306 to 2135) of each video.

2. We downsample the videos to 30 fps by getting rid half of the video frames.

REFERENCE: 
[1] Meta-rPPG: Remote Heart Rate Estimation Using a Transductive Meta-learner
"""

if __name__ == '__main__':
    demo()
