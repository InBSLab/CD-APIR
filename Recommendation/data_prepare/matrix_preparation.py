import pandas as pd
rt = pd.read_csv('D:\\sr-kl\\dataset\\ml-out\\rtmatrix.csv')
# Utility function: Map the value between [0,1]. The smaller the value, the better.
def ufm(a, amin=0.001, amax=0.349):
        return (amax - a) / (amax - amin)

for i in rt:
    rt[i] = rt[i].map(ufm)
# Calculate user satisfaction
def uf(i):
    if i < 0.5:
        return i * (i + 0.5)
    else:
        return i

for a in rt:
    rt[a]=rt[a].map(uf)
rt = rt*5
rt.to_csv('D:\\sr-kl\\dataset\\ml-out\\rating.csv', index=False)
