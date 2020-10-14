import numpy as np
import pandas as pd

def s_rate_equalization(s_rate):
    """
    Averaging of ratings by user in the matrix
    :param s_rate: User rating matrix: pandas.DataFrame
    :return:       Averaged user rating matrix: pandas.DataFrame
                   The average of ratings by users: pandas.Series
    """
    s_rate_mean = s_rate.mean()
    return (s_rate - s_rate.mean()), s_rate_mean


def matrix_prepare(s_rate_equalized, energy_ratio=0.6):
    """
     Preparing the rating prediction matrix: using SVD
    :param s_rate_equalized:  Averaged user rating matrix
    :param energy_ratio: The proportion of energy that to be retained: floating point
    :return: Predicted user rating
    """
    U, Sigma, VT = np.linalg.svd(s_rate_equalized.fillna(0).values)
    k = 0
    for i in range(len(Sigma)):
        if (Sigma[:i + 1] ** 2).sum() / (Sigma ** 2).sum() >= energy_ratio:
            k = i
            break
    NewData = U[:, :k] * np.mat(np.eye(k) * Sigma[:k]) * VT[:k, :]
    ND_DF = pd.DataFrame(NewData, index=s_rate_equalized.index,
                         columns=s_rate_equalized.columns)
    # Adding the original rating matrix
    s_rate_predict = ND_DF
    return s_rate_predict

if __name__ == '__main__':
    ENERGY_RATIO = 0.6

    s_rate_old = pd.read_csv('D:\\sr-kl\\dataset\\test\\s_rate_old1.0.csv')
    s_rate_old = s_rate_old.set_index('ServiceID')
    s_rate_old.rename(columns=int, inplace=False)
    s_rate_new = pd.read_csv('D:\\sr-kl\\dataset\\test\\s_rate_new0.8.csv')
    s_rate_new = s_rate_new.set_index('ServiceID')
    s_rate_new.rename(columns=int, inplace=False)

    s_rate_old_equalized, s_rate_old_mean = s_rate_equalization(s_rate_old)
    s_rate_svd_old = matrix_prepare(s_rate_old_equalized, ENERGY_RATIO)
    s_rate_new_equalized, s_rate_new_mean = s_rate_equalization(s_rate_new)
    s_rate_svd_new = matrix_prepare(s_rate_new_equalized, ENERGY_RATIO)

    s_rate_svd_old.to_csv('D:\\sr-kl\\dataset\\test\\s_rate_svd_old.csv')
    s_rate_svd_new.to_csv('D:\\sr-kl\\dataset\\test\\s_rate_svd_new.csv')
