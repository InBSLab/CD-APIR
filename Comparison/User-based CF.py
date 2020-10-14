import random
import math
from operator import itemgetter
import time

def ReadData(file,data):
    ''' Accessing     Rating data
        :param file   Rating matrix
        :param data   List for storing scoring data
    '''
    for line in file :
        line = line.strip('\n')
        linelist = line.split("::")
        data.append([linelist[0],linelist[1]])

def SplitData(data, M, key, seed):
    ''' Dividing data into training set and test set
        :param data   List for storing training set and test set
        :param M      Dividing the data into M shares
        :param key    Select the NO. key share as test set
        :param seed   Random seed
        :return train Training set
        :return test  Test set
    '''
    test = dict ()
    train = dict ()
    random.seed(seed)
    for user,item in data:
        if random.randint(0,M) == key:
            if user in test:
                test[user].append(item)
            else:
                test[user] = []
        else:
            if  user in train:
                train[user].append(item)
            else:
                train[user] = []
    return train, test

def UserSimilarityOld(train):
    W = dict()
    for u in train.keys():
        W[u] = dict()
        for v in train.keys():
            if u == v:
                continue
            W[u][v]  = len(list(set(train[u]) & set(train[v])))
            W[u][v] /= math.sqrt(len(train[u]) * len(train[v]) * 1.0)
    return W

def ItemSimilarity(train):
    ''' Calculating Item Similarity
        :param train Training set
        :return W    Two-dimensional matrix for storing user similarity
    '''
    C = dict()
    N = dict()
    # Calculate the number of users for per two items
    for u, items in train.items():
        for i in items:
            if i not in N:
                N[i] = 0
            N[i] += 1
            for j in items:
                if i == j:
                    continue
                if i not in C :
                    C[i] = dict()
                if j not in C[i]:
                    C[i][j] = 0
                C[i][j] += 1

    W = dict()
    for i, related_items in C.items():
        for j, cij in related_items.items():
            if i not in W :
                W[i] = dict()
            W[i][j] = cij / math.sqrt(N[i] * N[j])

    return W

def GetRecommendation(user, train ,W, N, K):
    ''' Recommender Results
        :param user  User
        :param train Training set
        :param W     Two-dimensional matrix for storing user similarity
        :param N     Number of recommended items
        :param K     Number of nearest neighbours selected
    '''
    rank = dict()
    ru = train[user]
    for i in  ru:
        for j,wj in sorted(W[i].items(), key=itemgetter(1),\
            reverse = True)[0:K]:
            if j in ru:
                continue
            if j in rank:
                rank[j] += wj
            else:
                rank[j] = 0

    rank = sorted(rank.items(), key=itemgetter(1), reverse = True)[0:N]
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
                    hit+= 1
            all += len(tu)
    print(hit)
    print(all)
    return hit/(all * 1.0)

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
                    hit+= 1
            all += N
    print(hit)
    print(all)
    return hit/(all * 1.0)
    
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

    print( 'len: ',len(recommned_items),'\n')
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
     train,test = SplitData(data, M, key, seed)
     W = ItemSimilarity(train)
     rec_start = time.time() Timing
     recall = Recall(train, test, W, N, K)
     precision = Precision(train, test, W, N, K)
     coverage = Coverage(train, test, W, N, K)
     rec_end = time.time()  # Timing
     rec_time_sum += rec_end - rec_start  # Timing
     rec_time = rec_time_sum / len(train.columns)  # Timing
     print( 'recall: ',recall,'\n')
     print( 'precision: ',precision,'\n')
     #print( 'Popularity: ',popularity,'\n')
     print( 'coverage: ', coverage,'\n')
     print('Average Recommend Time: %s' % rec_time)
else :
     print("this is not the main function")
