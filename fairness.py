import argparse
import numpy as np
from collections import defaultdict

from utils import load_recs, load_rels, load_features, get_prod, load_users

def mad_ranking(recs, features, rels, users, cutoff=100):
    item_clustering = {item:feature  for feature, sublist in features.items() for item in sublist}
    all_sum = defaultdict(lambda: 0)
    n_items = defaultdict(lambda: 0)
    item_count = {}
    item_gain = {}

    for u, u_r in recs.items():
        if u not in users:
            continue
        curr_rels = rels[u]
        if len(curr_rels):
            for i in u_r[:cutoff]:
                item_count[i] = item_count.get(i, 0) + 1
                if i in curr_rels:
                    item_gain[i] = item_gain.get(i, 0) + curr_rels[i]

    for item, gain in item_gain.items():
        v = gain/item_count[item]
        cluster = item_clustering.get(item, None)

        if cluster is not None:
            all_sum[cluster] += v
            n_items[cluster] += 1

    differences = []
    for i, feat1 in enumerate(features):
        feat1_avg = 0
        if n_items[feat1] > 0:
            feat1_avg = all_sum[feat1]/n_items[feat1]
        for j, feat2 in enumerate(features):
            if j > i+1:
                feat2_avg = 0
                if n_items[feat2] > 0:
                    feat2_avg = all_sum[feat2]/n_items[feat2]
                differences.append(abs(feat1_avg - feat2_avg))
    return np.average(differences)

def reo(train, recs, features, rels, users, cutoff=100):
    num = defaultdict(lambda: 0)
    den = defaultdict(lambda: 0)
    for u, u_r in recs.items():
        if u not in users:
            continue
        curr_rels = rels[u]
        if len(curr_rels):
            recommended_items = set([i for i in u_r[:cutoff] if i in curr_rels])
            for i, i_list in features.items():
                i_set = set(i_list)
                num[i] += len(recommended_items & i_set)
                den[i] += len((i_set  & set(curr_rels.keys())) - set(train[u]))

    PR = []
    for i in num: 
        if den[i] > 0:
            PR.append(num[i] / den[i])
        else:
            PR.append(0)
    metric_objs_list = np.std(PR) / np.mean(PR)
    return metric_objs_list


def rsp(train, recs, features, users, cutoff=100):
    num = defaultdict(lambda: 0)
    den = defaultdict(lambda: 0)
 
    for u, u_r in recs.items():
        if u not in users:
            continue
        recommended_items = set([i for i in u_r[:cutoff]])
        for i, i_list in features.items():
            i_set = set(i_list)
            num[i] += len(recommended_items & i_set)
            den[i] += len(i_set - set(train[u]))
    PR = []
    for i in num: 
        if den[i] > 0:
            PR.append(num[i] / den[i])
        else:
            PR.append(0)
    metric_objs_list = np.std(PR) / np.mean(PR)
    return metric_objs_list

def gamma_exposure(recs, features, users, gamma=0.5):
    item_clustering = defaultdict(lambda: [])
    for feature, sublist in features.items():
        for item in sublist:
            item_clustering[item].append(feature)

    ret = defaultdict(lambda: 0)
    for u, u_r in recs.items():
        if u not in users:
            continue
        curr_user = defaultdict(lambda: 0)
        for j, i in enumerate(u_r):
            if i in item_clustering:
                for cluster in item_clustering[i]:
                    curr_user[cluster] += gamma**(j)

        for cluster in curr_user:
            ret[cluster] += (1-gamma)*curr_user[cluster]
    for cluster in ret:
        ret[cluster] /= len(recs)
    return ret

def disparate_exposure(recs, features, users, total_items=3040):
    """ https://dl.acm.org/doi/pdf/10.1145/3404835.3463235 """
    # similar: https://link.springer.com/article/10.1007/s11257-021-09294-8
    item_clustering = {item:feature  for feature, sublist in features.items() for item in sublist}
    ret = 0
    for u, u_r in recs.items():
        if u not in users:
            continue
        curr_user = 0
        for j, i in enumerate(u_r):
            if i in item_clustering:
                curr_user += 1 / np.log2(2 + j)

        # Other alternative normalizations can be used here: len(u_r)
        tot_sum = sum([1/np.log2(2+i) for i in range(len(u_r))])
        ret += curr_user / tot_sum 
    ret /= len(recs)
    ret -= len(item_clustering) / total_items
    return ret

def proportion(recs, features, users, cutoff=100):
    ret = defaultdict(lambda: 0)
    for u, u_r in recs.items():
        if u not in users:
            continue
        recommended_items = set([i for i in u_r[:cutoff]])
        for i, i_list in features.items():
            i_set = set(i_list)
            ret[i] += len(recommended_items & i_set) / cutoff
    for i in ret:
        ret[i] /= len(recs)
    return ret

def delta_proportions(recs, features, users, cutoff=100):
    """ https://arxiv.org/pdf/2108.05152v1.pdf """
    eps = 1e-30
    p = proportion(recs, features, users, cutoff)
    ip = 1/ len(features)
    diff, diff_abs, diff_sq, diff_kl = 0,0,0,0
    for g in features:
        diff += ip - p[g]
        diff_abs += np.abs(ip - p[g])
        diff_sq += (ip - p[g])**2
        diff_kl += ip*np.log(ip/(p[g]+eps))
    return diff, diff_abs, diff_sq, diff_kl

def delta_exposure(recs, features, users, gamma=0.5):
    """ https://arxiv.org/pdf/2108.05152v1.pdf """
    eps = 1e-30
    p = gamma_exposure(recs, features, users, gamma)
    ip = (1-gamma)*sum([(1/16)*gamma**(i) for i in range(100)])
    diff, diff_abs, diff_sq, diff_kl = 0,0,0,0
    for g in features:
        diff += ip - p[g]
        diff_abs += np.abs(ip - p[g])
        diff_sq += (ip - p[g])**2
        diff_kl += ip*np.log(ip/(p[g]+eps))
    return diff, diff_abs, diff_sq, diff_kl


if __name__ == '__main__':
    # Input one run file, features file and relevant (qrels)

    parser = argparse.ArgumentParser(description='Compute MAD.')
    parser.add_argument('-rec', help='rec file')
    parser.add_argument('-feat', help='features file')
    parser.add_argument('-rel', help='relevance file')
    parser.add_argument('-train', help='train file')
    parser.add_argument('-users', help='reduced users file')

    args = parser.parse_args()
    features = load_features(args.feat)
    recs = load_recs(args.rec)
    rels = load_rels(args.rel)
    train = load_recs(args.train)

    if args.users:
        users = load_users(args.users)
    else:
        users = recs

    diff, diff_abs, diff_sq, diff_kl = delta_proportions(recs, features, users)
    print ("Prop_delta_diff", diff)
    print ("Prop_delta_abs", diff_abs)
    print ("Prop_delta_sq", diff_sq)
    print ("Prop_delta_kl", diff_kl)

    disp_exp = disparate_exposure(recs, features, users)
    print ("Disp_exp", disp_exp)
   
    diff, diff_abs, diff_sq, diff_kl = delta_exposure(recs, features, users)
    print ("Exp_delta_diff", diff)
    print ("Exp_delta_abs", diff_abs)
    print ("Exp_delta_sq", diff_sq)
    print ("Exp_delta_kl", diff_kl)
 
    ret_mad =  mad_ranking(recs, features, rels, users)
    print ("MAD", ret_mad)
    ret_reo = reo(train, recs, features, rels, users)
    print ("REO", ret_reo)
    ret_rsp = rsp(train, recs, features, users)
    print ("RSP", ret_rsp)

