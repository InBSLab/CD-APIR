import pandas as pd
import time

def s_rate_equalization(s_rate):
    """
    Averaging ratings in the matrix
    :param s_rate: User rating matrix: pandas.DataFrame
    :return:       Averaged user rating matrix: pandas.DataFrame
    """
    s_rate_mean = s_rate.mean()
    return (s_rate - s_rate.mean())

def recommend_svd(s_rate_equalized, s_rate_predict, u, num=10):
    """
    Generating a recommendation list
    :param s_rate_equalized:   Averaged user rating matrix
    :param s_rate_predict:     Predicted user rating matrix
    :param u:                  User ID: integer 
    :param num:                Maximum number of services in the list: integer 
    :return:Recommended list: list. The items are tuple. Ttems within the tuple are service ID and service's rating.
    """
    # Selecting services that were not rated in the original rating matrix and have a predictive value at or above the average score
    r_1 = s_rate_predict.loc[s_rate_equalized[u].isnull(), u][
        s_rate_predict[u] >= 0].sort_values(ascending=False)
    rec_movies = [([i, r_1[i]]) for i in r_1[:num].index]
    return rec_movies

if __name__ == '__main__':
    #RECOMMEND_NUM = 10

    s_rate_ceshiji = pd.read_csv('D:\\sr-kl\\dataset\\data-preparation\\s_rate_ceshiji.csv')
    s_rate_ceshiji = s_rate_ceshiji.set_index('ServiceID')
    s_rate_ceshiji.rename(columns=int, inplace=False)
    s_rate_predict = pd.read_csv('D:\\sr-kl\\dataset\\predict\\js.csv')
    s_rate_predict = s_rate_predict.set_index('ServiceID')
    s_rate_predict.rename(columns=int, inplace=False)
    s_rate_xunlianji = pd.read_csv('D:\\sr-kl\\dataset\\data-preparation\\s_rate_xunlianji.csv')
    s_rate_xunlianji = s_rate_xunlianji.set_index('ServiceID')
    s_rate_xunlianji.rename(columns=int, inplace=False)

    s_rate_predict_equalized = s_rate_equalization(s_rate_xunlianji)

    # Precision & Recall & Coverage & Diversity

    r_and_s_sum = 0
    r_sum = 0
    s_sum = 0
    ru_set = set()
    sum_diversity_u = 0
    user_minus = 0
    rec_time_sum = 0  # Timing
    for u in s_rate_xunlianji.columns:  # Printing column
        print('\r%s' % u, end='', flush = True)
        select_set = set(s_rate_ceshiji[~s_rate_ceshiji[u].isnull()][u].index) #打印用户u选择的服务的序列名
        rec_start = time.time()  # Timing
        recommend_list_with_score = recommend_svd(s_rate_predict_equalized,
                                                  s_rate_predict, u)
        rec_end = time.time()  # Timing
        recommend_list = [i[0] for i in recommend_list_with_score]
        recommend_set = set(recommend_list)
        r_and_s_sum += len(recommend_set & select_set)
        r_sum += len(recommend_set)
        s_sum += len(select_set)

        for i in recommend_list:
            ru_set.add(i)
        rec_time_sum += rec_end - rec_start  # Timing
    coverage = len(ru_set) / len(s_rate_xunlianji.index)
    rec_time = rec_time_sum / len(s_rate_xunlianji.columns)  # Timing
    precision = r_and_s_sum / r_sum
    recall = r_and_s_sum / s_sum
    f_measure = (2 * precision * recall) / ( precision + recall)
    print('Average Recommend Time: %s' % '\n',rec_time)  # Timing
    print('Precision: %s' % '\n',precision)
    print('Recall: %s' % '\n',recall)
    print('F-Measure: %s' % '\n',f_measure)
    print('Coverage: %s' % '\n',coverage)
