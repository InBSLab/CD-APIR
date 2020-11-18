from numpy import linspace
from scipy.stats import gaussian_kde
import pandas as pd
import scipy.stats

s_rate_svd_old = pd.read_csv('D:\\sr-kl\\dataset\\test\\s_rate_svd_old.csv')
s_rate_svd_new = pd.read_csv('D:\\sr-kl\\dataset\\test\\s_rate_svd_new.csv')

# Calculating JS divergence
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

def run_normal_unimodal(p,q):
    js_div = JS_divergence(p, q)
    print(js_div)

def main():
    for key in range(339):
        run_normal_unimodal(s_rate_svd_old[str(key)], s_rate_svd_new[str(key)])
        
if __name__ == "__main__":
    main()
