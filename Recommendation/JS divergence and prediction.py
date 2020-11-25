from numpy import linspace
from scipy.stats import gaussian_kde
import pandas as pd
import scipy.stats
import numpy as np

s_rate_svd_old = pd.read_csv('D:\\sr-kl\\dataset\\predict\\s_rate_svd_old.csv')
s_rate_svd_new = pd.read_csv('D:\\sr-kl\\dataset\\predict\\s_rate_svd_new.csv')

def JS_divergence(p, q):
    n1 = len(p)
    n2 = len(q)
    if n1 == 0 or n2 == 0:
        return 0
    # Estimate the PDFs using Gaussian KDE
    pdf1 = gaussian_kde(p)
    pdf2 = gaussian_kde(q)

    # Calculate the interval to be analyzed further
    a = min(min(p), min(q))
    b = max(max(p), max(q))

    lin = linspace(a, b, max(n1, n2))
    p = pdf1.pdf(lin)
    q = pdf2.pdf(lin)
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)

def ufm(a, amin, amax):
    return (a - amin) / (amax - amin)

if __name__ == "__main__":
    js = pd.Series([JS_divergence(s_rate_svd_old[str(key)], s_rate_svd_new[str(key)]) for key in range(339)]) #Calculat JS divergence
    amin = min(js)
    amax = max(js)
    js = ufm(js, amin, amax) #Normalize JS divergence
    js = pd.DataFrame(js)
    js = js.T  #Transpose JS divergence

    #Construct a predictive rating matrix
    users = pd.read_csv('D:\\sr-kl\\dataset\\data-preparation\\users.csv') #Make column names (building the first row of the matrix)
    wslist = pd.read_csv('D:\\sr-kl\\dataset\\data-preparation\\wslist.csv') #Make row names (to build the first column of the matrix)
    s_rate_predict = pd.DataFrame(index=wslist.ServiceID, dtype=np.float)

    #theta = 0.1
    for i in range(339):
        if js.at[0, i] > 0.9:
            # User preferences drift dramatically. Recommendation should be based on s_rate_svd_new.
            s_rate_predict[str(i)] = s_rate_svd_new[str(i)]
            print(s_rate_predict[str(i)])
        elif (js.at[0, i] >= 0.1 ) and (js.at[0, i] <= 0.9 ):
            # Making a trade-off to combine s_rate_svd_old and s_rate_svd_new.
            s_rate_predict[str(i)] = s_rate_svd_new[str(i)] * js.at[0, i] + s_rate_svd_old[str(i)] * (1 - js.at[0, i])
            print(s_rate_predict[str(i)])
        else:
            # It implies a minor drift. s_rate_svd_old may be more accurate.
            s_rate_predict[str(i)] = s_rate_svd_old[str(i)]
            print(s_rate_predict[str(i)])

    print("End")
    print(s_rate_predict)
    s_rate_predict.to_csv('D:\\sr-kl\\dataset\\predict\\js.csv')
    
