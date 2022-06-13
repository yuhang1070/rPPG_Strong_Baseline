from sklearn.metrics import mean_absolute_error as compute_mae
from sklearn.metrics import mean_squared_error as __mse
from scipy.stats import pearsonr
import numpy as np


def compute_rmse(pred, label):
    return np.sqrt(__mse(pred, label))


def compute_r(pred, label):
    return pearsonr(pred, label)[0]


def compute_hr_metric(pred, label):
    pred = np.array(pred, np.float32)
    label = np.array(label, np.float32)

    sd = np.std(pred - label)
    mae = compute_mae(pred, label)
    rmse = compute_rmse(pred, label)
    r = compute_r(pred, label)

    return sd, mae, rmse, r

def cal_hr_metric(hr_gt, hr_es, ):
    hr_gt = np.array(hr_gt, dtype=np.float32)
    hr_es = np.array(hr_es, dtype=np.float32)
    hr_err = hr_gt - hr_es

    STD = np.std(hr_err)
    MAE = np.mean(np.abs(hr_err))
    RMSE = np.sqrt(np.mean(np.square(hr_err)))
    R, _ = pearsonr(hr_es, hr_gt)

    ME = np.mean(hr_err)
    MER = np.mean(np.abs(hr_err) / hr_gt)

    return {
        'STD': float(STD),
        'MAE': float(MAE),
        'RMSE': float(RMSE),
        'R': float(R),
        'ME': float(ME),
        'MER': float(MER),
    }


if __name__ == '__main__':
    x_gt = [1, 2, 3, 4]
    x_pred = [1.4, 2.4, 6, 3.5]

    print('sd, mae, rmse, r: {}'.format(compute_hr_metric(pred=x_pred, label=x_gt)))
