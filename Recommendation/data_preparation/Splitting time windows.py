# Splitting time windows

import pandas as pd
import numpy as np

ratings = pd.read_csv('D:\\sr-kl\\dataset\\test\\ratings_xunlianji.csv')
users = pd.read_csv('D:\\sr-wsdream\\data\\tempstart\\users.csv')
wslist = pd.read_csv('D:\\sr-wsdream\\data\\tempstart\\wslist.csv')
s_rate = pd.DataFrame(columns=users.UserID, index=wslist.ServiceID,
                      dtype=np.float)
rate_time = pd.DataFrame(columns=users.UserID, index=wslist.ServiceID,
                         dtype=np.int)
s_rate = s_rate.fillna(0) + ratings.pivot(index='ServiceID', columns='UserID',
                                          values='Rating')
rate_time = rate_time.fillna(0) + ratings.pivot(index='ServiceID',
                                                columns='UserID',
                                                values='Timestamp')
ratings_new = ratings.copy()
ratings_old = ratings.copy()
for u in users.UserID:
    ratings_new[(ratings_new['UserID'] == u) & (ratings_new['Timestamp'] < ratings_new[ratings_new['UserID'] == u]['Timestamp'].quantile(0.8))] = np.nan
    ratings_old[(ratings_old['UserID'] == u) & (ratings_old['Timestamp'] > ratings_old[ratings_old['UserID'] == u]['Timestamp'].quantile(1.0))] = np.nan
s_rate_new = s_rate.copy()
s_rate_new[(rate_time < rate_time.quantile(0.8))] = np.nan
s_rate_old = s_rate.copy()
s_rate_old[(rate_time > rate_time.quantile(1.0))] = np.nan

s_rate_new.to_csv('D:\\sr-kl\\dataset\\test\\s_rate_new0.8.csv')
s_rate_old.to_csv('D:\\sr-kl\\dataset\\test\\s_rate_old1.0.csv')
