# rPPG Strong Baseline
***An open source toolbox and Strong Baseline for rPPG-based remote physiological measurement.***

Official code for https://arxiv.org/abs/2206.05687

For more information, https://arxiv.org/abs/2206.05687

## train 

|  train   | Application  |
|  ----  | ----  |
| train2.py          | train |
| valid2.py          | valid |
| make_dataloader.py | Make train/valid dataloader, and data enhancement |
| load_data.py | Load data from path profile |
| load_path.py | Load train/valid path profile |

## STMap

|  STMap   | Application  |
|  ----  | ----  |
| *.png              | Examples for STMap generated by our method (https://arxiv.org/abs/2206.05687) |

## preprocess

|  preprocess   | Application  |
|  ----  | ----  |
| preprocess_data.py  | Train/Test Data Generation. |
| preprocess_crop.py | Face cropping, and generate STMap from face videos. |
| **gen_STmap.py** | **STMap Generation, Random Patch Cropping** |
| preprocess_landmark.py | Face landmarking |
| preprocess_detect.py   | Face detection |

## models

|  models   | Application  |
|  ----  | ----  |
|  **base/net_helper.py**      |    **Base model**   |

## tools

|  tools   | Application  |
|  ----  | ----  |
| eval_tools.py  | For heart rate prediction, standard deviation of the error (Std), mean absolute error (MAE), root mean square error (RMSE), mean error rate (MER) and Pearson’s correlation coefficient (r) are employed for performance evaluation. |
| face_tools.py  | Face cropping and segmentation. |
| image_tools.py | Color space transformation. |
| net_tools.py | Network tools. |
| io_tools.py    | Read and write file function library. |
| log_tools.py   | Log. |
| meter_tools.py | AverageMeter. |
| metric_tools.py | Calculate IBI error. |
| ppg_tools.py | Calculate heart rate from PPG/rPPG signal. |
| video_tools.py | Video processing. |
| skin_tools.py | Skin segmentation. |
| sig_tools.py  | Signal filtering, detrending. |

## others

|  Filename   | Application  |
|  ----  | ----  |
| gen_path/gen_vipl.py | Generate path profile for VIPL database |
| gen_path/gen_ubfc.py | Generate path profile for UBFC-rPPG database |
| gen_path/gen_mahnob.py | Generate path profile for MAHNOB-HCI database |
| datasets/mahnob.py | MAHNOB HCI database processing library. |
| datasets/pure.py   | PURE database processing library. |
| datasets/ubfc2.py  | UBFC-rPPG database processing library. |
| datasets/vipl.py   | VIPL-HR database processing library. |
| losses/negative_pearson_loss.py | Negative pearson loss function. |
| losses/cross_snr_loss.py | Signal-to-noise loss function. |
| config/parameters.py | Configuration. |


***To be continued!***
