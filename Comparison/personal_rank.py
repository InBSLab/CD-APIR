import numpy as np
import pandas as pd
import time

def s_rate_equalization(s_rate):
    """
    Averaging ratings in the matrix
    :param s_rate: User rating matrix: pandas.DataFrame
    :return:       Averaged user rating matrix: pandas.DataFrame
    """
    s_rate_mean = s_rate.mean()
    return (s_rate - s_rate.mean()), s_rate_mean


def graph_gen(s_rate, rate_limit=1.97):
    """
    Generating bipartite graph
    :param s_rate:     User rating matrix
    :param rate_limit: The link will be considered only if the rating is greater than this value.
    :return: Bipartite graph: matrix
             The name of the value on the coordinates axis.
    """
    user_num = len(s_rate.columns)
    vertex = ['U_%s' % i for i in s_rate.columns]
    vertex.extend(['S_%s' % i for i in s_rate.index])
    graph = np.zeros((len(vertex), len(vertex)))
    for ii, i in enumerate(s_rate.index):
        for ji, j in enumerate(s_rate.columns):
            if s_rate.loc[i, j] >= rate_limit:
                graph[ii + user_num][ji] = 1
                graph[ji][ii + user_num] = 1
            print('\r%s\t%s' % (i, j), end='', flush=True)
    return graph, vertex


def M_gen(graph):
    """
    Generating matrix
    :param graph: bipartite graph
    :return: matrix
    """
    g = graph.copy().astype(np.float)
    return g


def matrix_prepare(s_rate, rate_limit= 1.97, alpha=0.8):
    """
    Matrix preparation, an encapsulation of the above two functions
    :param s_rate:     user rating matrix 
    :param alpha:      Probability of going upstream of a node to the next node
    :return:           PersonalRank 
    """
    graph, vertex = graph_gen(s_rate, rate_limit)
    M = M_gen(graph)
    r_all = np.linalg.inv(np.eye(M.shape[0]) - alpha * M.T)
    return r_all, vertex


def scores_prepare(r_all, vertex):
    """
    Sorting preparation
    :param r_all:  Sorting matrix
    :param vertex: The name of the value on the coordinates axis
    :return: Sorting list
    """
    scores = pd.DataFrame(r_all, index=vertex, columns=vertex)
    scores = scores[scores.index.str.startswith('U_')]
    scores = scores.T[scores.T.index.str.startswith('S_')]
    scores.rename(index=lambda x: int(x[2:]), columns=lambda x: int(x[2:]),
                  inplace=True)
    return scores


def personal_rank(scores, u, num=10):
    """
    Recommendations based on a sorting list
    :param scores: Sorting list
    :param u:      User ID
    :param num:    Number of items
    :return: Recommender list: Each item is a tuple, including service ID and sorted by PR value
    """
    scores_u = scores[u].sort_values(ascending=False)
    rec_list = [([i, scores_u[i]]) for i in scores_u[:num].index]
    return rec_list


if __name__ == '__main__':

    MATRIX_PATH = 'D:\\sr-kl\\dataset\\'

    s_rate_old = pd.read_csv(MATRIX_PATH + '/s_rate_xunlianji.csv')
    s_rate_old = s_rate_old.set_index('ServiceID')
    s_rate_old.rename(columns=int, inplace=True)

    s_rate_old_equalized, s_rate_old_mean = s_rate_equalization(s_rate_old)

    model_start = time.time()  # Timing
    r, v = matrix_prepare(s_rate_old_equalized)
    scores = scores_prepare(r, v)
    model_end = time.time()  # Timing
    recommend_list_example = personal_rank(scores, 5)
    print(recommend_list_example)
    print('Model time: %s' % (model_end - model_start))  # Timing

    # Precision & Recall & Coverage & Diversity
    s_rate_new = pd.read_csv(MATRIX_PATH + '/s_rate_ceshiji.csv')
    s_rate_new = s_rate_new.set_index('ServiceID')
    s_rate_new.rename(columns=int, inplace=True)

    r_and_s_sum = 0
    r_sum = 0
    s_sum = 0
    ru_set = set()
    sum_diversity_u = 0
    user_minus = 0
    rec_time_sum = 0  # Timing
    for u in s_rate_old.columns:
        print('\r%s' % u, end='', flush=True)
        select_set = set(s_rate_new[~s_rate_new[u].isnull()][u].index)
        rec_start = time.time()  # Timing
        recommend_list_with_score = personal_rank(scores, u)
        rec_end = time.time()  # Timing
        recommend_list = [i[0] for i in recommend_list_with_score]
        recommend_set = set(recommend_list)
        r_and_s_sum += len(recommend_set & select_set)
        r_sum += len(recommend_set)
        s_sum += len(select_set)
        for i in recommend_list:
            ru_set.add(i)
        rec_time_sum += rec_end - rec_start  # Timing
    coverage = len(ru_set) / len(s_rate_old.index)
    rec_time = rec_time_sum / len(s_rate_old.columns)  # Timing
    precision = r_and_s_sum / r_sum
    recall = r_and_s_sum / s_sum
    f_measure = (2 * precision * recall) / (precision + recall)
    print('Average Recommend Time: %s' % rec_time)  # Timing
    print('Precision: %s' % precision)
    print('Recall: %s' % recall)
    print('F-Measure: %s' % f_measure)
    print('Coverage: %s' % coverage)
