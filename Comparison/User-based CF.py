import random
import math
from operator import itemgetter
import time


def ReadData(file, data):
    ''' Accessing     Rating data
        :param file   Rating matrix
        :param data   List for storing scoring data
    '''
    for line in file:
        line = line.strip('\n')
        linelist = line.split("::")
        data.append([linelist[0], linelist[1]])


def SplitData(data, M, key, seed):
    ''' Dividing data into training set and test set
        :param data   List for storing training set and test set
        :param M      Dividing the data into M shares
        :param key    Select the NO. key share as test set
        :param seed   Random seed
        :return train Training set
        :return test  Test set
    '''
    test = dict()
    train = dict()
    random.seed(seed)
    for user, item in data:
        if random.randint(0, M) == key:
            if user in test:
                test[user].append(item)
            else:
                test[user] = []
        else:
            if user in train:
                train[user].append(item)
            else:
                train[user] = []
    return train, test


def UserSimilarity(train):
    ''' Calculating Item Similarity
        :param train Training set
        :return W    Two-dimensional matrix for storing user similarity
    '''
    # Create an item-to-user regression table to reduce the time complexity of calculating user similarity
    item_users = dict()
    for u, items in train.items():
        for i in items:
            if (i not in item_users):
                item_users[i] = set()
            item_users[i].add(u)
        C = dict()
        N = dict()
        # Calculate the number of items shared between each users
        for i, users in item_users.items():
            for u in users:
                if (u not in N):
                    N[u] = 1
                N[u] += 1
                for v in users:
                    if u == v:
                        continue
                    if (u not in C):
                        C[u] = dict()
                    if (v not in C[u]):
                        C[u][v] = 0
                    C[u][v] += (1 / math.log(1 + len(users)))
    W = dict()
    for u, related_users in C.items():
        for v, cuv in related_users.items():
            if (u not in W):
                W[u] = dict()
            # Using cosine similarity to calculate similarity between users
            W[u][v] = cuv / math.sqrt(N[u] * N[v])

    return W

def GetRecommendation(user, train, W, N, K):
    ''' Recommender Results
        :param user  User
        :param train Training set
        :param W     Two-dimensional matrix for storing user similarity
        :param N     Number of recommended items
        :param K     Number of nearest neighbours selected
    '''
    rank = dict()
    interacted_items = train[user]
    # Selecting K nearest neighbors to calculate the rating
    for v, wuv in sorted(W[user].items(), key=itemgetter(1), \
                         reverse=True)[0:K]:
        for i in train[v]:
            if i in interacted_items:
                continue
            if i in rank:
                rank[i] += wuv
            else:
                rank[i] = 0

    # Selecting the N items with the highest ratings as the recommender results.
    rank = sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    return rank


def Recall(train, test, W, N, K):
    ''' Calculating recall for recommended results
        :param train Training set
        :param test  Test set
        :param W     Two-dimensional matrix for storing user similarity
        :param N     Number of recommended items
        :param K     Number of nearest neighbours selected
    '''
    hit = 0
    all = 0
    for user in train.keys():
        if user in test:
            tu = test[user]
            rank = GetRecommendation(user, train, W, N, K)
            for item, pui in rank:
                if item in tu:
                    hit += 1
            all += len(tu)
    return hit / (all * 1.0)


def Precision(train, test, W, N, K):
    '''Calculating precision for recommended results
        :param train Training set
        :param test  Test set
        :param W     Two-dimensional matrix for storing user similarity
        :param N     Number of recommended items
        :param K     Number of nearest neighbours selected
    '''
    hit = 0
    all = 0
    for user in train.keys():
        if user in test:
            tu = test[user]
            rank = GetRecommendation(user, train, W, N, K)
            for item, pui in rank:
                if item in tu:
                    hit += 1
            all += N
    return hit / (all * 1.0)

def Coverage(train, test, W, N, K):
    '''Calculating coverage for recommended results
        :param train Training set
        :param test  Test set
        :param W     Two-dimensional matrix for storing user similarity
        :param N     Number of recommended items
        :param K     Number of nearest neighbours selected
    '''
    recommned_items = set()
    all_items = set()

    for user in train.keys():
        for item in train[user]:
            all_items.add(item)

        rank = GetRecommendation(user, train, W, N, K)
        for item, pui in rank:
            recommned_items.add(item)
    return len(recommned_items) / (len(all_items) * 1.0)


if __name__ == '__main__':
    data = []
    M = 10
    key = 10
    seed = 1
    N = 10
    K = 1
    W = dict()
    rank = dict()
    rec_time_sum = 0

    print("this is the main function")
    file = open('D:\\sr-kl\\dataset\\ml-out\\ratings.dat')
    ReadData(file, data)
    train, test = SplitData(data, M, key, seed)
    W = UserSimilarity(train)
    rec_start = time.time()
    precision = Precision(train, test, W, N, K)
    print('precision: ', precision, '\n')
    recall = Recall(train, test, W, N, K)
    print(    'recall: ', recall, '\n')
    coverage = Coverage(train, test, W, N, K)
    print('coverage: ', coverage, '\n')
    rec_end = time.time()  # Timing
    rec_time_sum += rec_end - rec_start  # Timing
    rec_time = rec_time_sum / len(train)  # Timing
    print('Average Recommend Time: %s' % rec_time)
else:
    print("this is not the main function")
