import numpy as np
import pandas as pd

# Splitting training set and test set
ratings = pd.read_csv('DD:\\sr-kl\\dataset\\ml-out\\ratings.csv')
users = pd.read_csv('D:\\sr-wsdream\\data\\tempstart\\users.csv')
wslist = pd.read_csv('D:\\sr-wsdream\\data\\tempstart\\wslist.csv')
s_rate = pd.DataFrame(columns=users.UserID, index=wslist.ServiceID,
                      dtype=np.float)
rate_time = pd.DataFrame(columns=users.UserID, index=wslist.ServiceID,
                         dtype=np.int)
rate_time = rate_time.fillna(0) + ratings.pivot(index='ServiceID',
                                                columns='UserID',
                                                values='Timestamp')
s_rate = s_rate.fillna(0) + ratings.pivot(index='ServiceID', columns='UserID',
                                          values='Rating')
ratings_new = ratings.copy()
ratings_old = ratings.copy()
for u in users.UserID:
    ratings_new[(ratings_new['UserID'] == u) & (ratings_new['Timestamp'] < ratings_new[ratings_new['UserID'] == u]['Timestamp'].quantile(0.9))] = np.nan
    ratings_old[(ratings_old['UserID'] == u) & (ratings_old['Timestamp'] > ratings_old[ratings_old['UserID'] == u]['Timestamp'].quantile(0.9))] = np.nan
ratings_new.dropna(inplace=True)
ratings_old.dropna(inplace=True)
s_rate_new = s_rate.copy()
s_rate_new[(rate_time < rate_time.quantile(0.9))] = np.nan
s_rate_old = s_rate.copy()
s_rate_old[(rate_time > rate_time.quantile(0.9))] = np.nan
rate_time_new = rate_time.copy()
rate_time_new[s_rate_new.isnull()] = np.nan
rate_time_old = rate_time.copy()
rate_time_old[s_rate_old.isnull()] = np.nan
ratings_new.to_csv('D:\\sr-kl\\dataset\\test\\ratings_ceshiji.csv')
ratings_old.to_csv('D:\\sr-kl\\dataset\\test\\ratings_xunlianji.csv')
s_rate.to_csv('D:\\sr-kl\\dataset\\test\\s_rate.csv')
s_rate_new.to_csv('D:\\sr-kl\\dataset\\test\\s_rate_ceshiji.csv')
s_rate_old.to_csv('D:\\sr-kl\\dataset\\test\\s_rate_xunlianji.csv')
rate_time_new.to_csv('D:\\sr-kl\\dataset\\test\\rate_time_ceshiji.csv')
rate_time_old.to_csv('D:\\sr-kl\\dataset\\test\\rate_time_xunlianji.csv')
rate_time.to_csv('D:\\sr-kl\\dataset\\test\\rate_time.csv')
