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


if __name__ == '__main__':
    x_gt = [1, 2, 3, 4]
    x_pred = [1.4, 2.4, 6, 3.5]

    print('sd, mae, rmse, r: {}'.format(compute_hr_metric(pred=x_pred, label=x_gt)))
