# rPPG_Strong_Baseline
***An open source toolbox and Strong Baseline for rPPG-based remote physiological measurement.***

For more information, https://arxiv.org/abs/2206.05687

|  Filename   | Application  |
|  ----  | ----  |
| preprocess/preprocess_landmark.py | Face landmarking |
| preprocess/preprocess_detect.py   | Face detection |
| tools/eval_tools.py  | For heart rate prediction, standard deviation of the error (Std), mean absolute error (MAE), root mean square error (RMSE), mean error rate (MER) and Pearson’s correlation coefficient (r) are employed for performance evaluation. |
| tools/face_tools.py  | Face cropping and segmentation. |
| tools/image_tools.py | Color space transformation. |
| tools/io_tools.py    | Read and write file function library. |
| tools/log_tools.py   | Log. |
| tools/meter_tools.py | AverageMeter. |
| tools/metric_tools.py | Calculate IBI error. |
| tools/ppg_tools.py | Calculate heart rate from PPG/rPPG signal. |
| tools/video_tools.py | Video processing. |
| tools/skin_tools.py | Skin segmentation. |
| tools/sig_tools.py  | Signal filtering, detrending. |
| datasets/mahnob.py | MAHNOB HCI database processing library. |
| datasets/pure.py   | PURE database processing library. |
| datasets/ubfc2.py  | UBFC-rPPG database processing library. |
| datasets/vipl.py   | VIPL-HR database processing library. |
| losses/negative_pearson_loss.py | Negative pearson loss function. |
| losses/cross_snr_loss.py | Signal-to-noise loss function. |
| config/parameters.py | Configuration. |

***To be continued!***
