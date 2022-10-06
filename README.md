# rPPG_Strong_Baseline
***An open source toolbox and Strong Baseline for rPPG-based remote physiological measurement.***

## tools
|  Filename   | Application  |
|  ----  | ----  |
| eval_tools.py  | For heart rate prediction, standard deviation of the error (Std), mean absolute error (MAE), root mean square error (RMSE), mean error rate (MER) and Pearson’s correlation coefficient (r) are employed for performance evaluation. |
| face_tools.py  | Face cropping and segmentation. |
| image_tools.py | Color space transformation. |
| io_tools.py    | Read and write file function library. |
| log_tools.py   | Log. |
| meter_tools.py | AverageMeter. |
| metric_tools.py | Calculate IBI error. |
| ppg_tools.py | Calculate heart rate from PPG/rPPG signal. |
| video_tools.py | Video processing. |
| skin_tools.py | Skin segmentation. |
| sig_tools.py  | Signal filtering, detrending. |

## datasets
|  Filename   | Application  |
|  ----  | ----  |
| mahnob.py | MAHNOB HCI database processing library. |
| pure.py   | PURE database processing library. |
| ubfc2.py  | UBFC-rPPG database processing library. |
| vipl.py   | VIPL-HR database processing library. |

## losses
|  Filename   | Application  |
|  ----  | ----  |
| negative_pearson_loss.py | Negative pearson loss function. |
| cross_snr_loss.py | Signal-to-noise loss function. |

## config
|  Filename   | Application  |
|  ----  | ----  |
| parameters.py | Configuration. |

***To be continued!***
