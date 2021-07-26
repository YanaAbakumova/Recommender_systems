"""
Metrics

"""
import numpy as np

def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return (flags.sum() > 0) * 1

def hit_rate_at_k(recommended_list, bought_list, k=5):
    return hit_rate(recommended_list[:k], bought_list)

def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(recommended_list)

def precision_at_k(recommended_list, bought_list, k=5):
    return precision(recommended_list[:k], bought_list)

def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    flags = np.isin(recommended_list, bought_list)
    return np.dot(flags, prices_recommended).sum() / prices_recommended.sum()

def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(bought_list)


def recall_at_k(recommended_list, bought_list, k=5):
    return recall(recommended_list[:k], bought_list)


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    prices_bought = np.array(prices_bought)
    flags = np.isin(recommended_list, bought_list)
    return np.dot(flags, prices_recommended).sum() / prices_bought.sum()


def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    recommended_list = recommended_list[recommended_list <= k]

    relevant_indexes = np.nonzero(np.isin(recommended_list, bought_list))[0]
    if len(relevant_indexes) == 0:
        return 0
    amount_relevant = len(relevant_indexes)


    sum_ = sum(
        [precision_at_k(recommended_list, bought_list, k=index_relevant + 1) for index_relevant in relevant_indexes])
    return sum_ / amount_relevant


def map_k(recommended_list, bought_list, k=5):
    num_users = len(bought_list)
    sum_ap_k = 0
    for i in range(num_users):
        b_list = np.array(bought_list[i])
        r_list = np.array(recommended_list[i])[:k]
        relevant_indexes = np.nonzero(np.isin(r_list, b_list))[0]
        if len(relevant_indexes) == 0:
            continue

        amount_relevant = len(relevant_indexes)
        sum_ = sum([precision_at_k(r_list, b_list, k=index_relevant + 1) for index_relevant in relevant_indexes])
        sum_ap_k += sum_ / amount_relevant

    return sum_ap_k / num_users

def ndcg_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    flags = np.isin(recommended_list, bought_list)
    discount_i = np.append([1, 2], [np.log2(i) for i in range(3, k+1)])
    dcg = 1/k * sum(flags / discount_i)
    ideal_dcg = 1/k * sum(np.ones(len(flags)) / discount_i)
    return dcg / ideal_dcg

def reciprocal_rank(recommended_list, bought_list, k=1):
    num_users = len(bought_list)
    k_u = []
    for i in range(num_users):
        b_list = np.array(bought_list[i])
        r_list = np.array(recommended_list[i])[:k]
        if sum(np.isin(r_list, b_list)) ==0:
            k_u.append(0)
            continue
        k_u.append(1 / (np.nonzero(np.isin(r_list, b_list))[0][0] +1))
    if k_u ==[]:
        return 0
    return np.array(k_u).mean()        