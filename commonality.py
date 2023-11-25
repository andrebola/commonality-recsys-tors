import os
import argparse
import csv
import json
import numpy as np
from utils import load_recs, load_rels, load_features, get_prod, load_users
from collections import defaultdict

def commonality_log(f1,features_file, total_items, users=None, gamma=0.95):
    features = load_features(features_file)
    recs = load_recs(f1)
    if users == None:
        users = recs
    ret = {}
    ret_linear = {}
    for feature in features.keys():
        ret[feature] = 0
        ret_linear[feature] = 0
    #items_list = {item:feature  for feature, sublist in features.items() for item in sublist}
    items_list = defaultdict(lambda: [])
    for feature, sublist in features.items():
        for item in sublist:
            items_list[item].append(feature)

    max_n_recs = len(list(recs.values())[0])
    p = np.array([(1-gamma)*(gamma)**(j-1) for j in range(1, total_items+1)])
    for user in recs:
        if user not in users:
            continue
        feat_user = {}
        for pos,i in enumerate(recs[user]):
            if i in items_list:
                for cat in items_list[i]:
                    if cat not in feat_user:
                        feat_user[cat] = []
                    feat_user[cat].append(pos)

        for feature in features.keys():
            m = len(features[feature])
            if feature not in feat_user:
                missing = m
                feat_user[feature] = []
            else:
                missing = m - len(feat_user[feature])

            for i in range(missing):
                feat_user[feature].append(max_n_recs+i)

            f_i_log = np.repeat(1.0, total_items)
            f_i = np.repeat(1.0, total_items)
            last_item = 0
            for k, i in enumerate(feat_user[feature]):
                for j in range(last_item,i):
                    f_i_log[j] = np.log(k+1)/np.log(m+1)
                    f_i[j] = k/float(m) # K is 0-index
                last_item = i

            #print (f_i[:200])

            #print (np.max(p*f_i))
            a = np.log(p*f_i) 
            #print (np.argmax(a))
            a_log = np.log(p*f_i_log) 
            b = np.max(a)
            b_log = np.max(a_log)
            curr_user = np.log(np.sum(np.exp(a-b))) + b
            curr_user_log = np.log(np.sum(np.exp(a_log-b_log))) + b_log
           
            ret_linear[feature] += curr_user
            ret[feature] += curr_user_log

    for feat in features.keys():
        print (feat, ret_linear[feat])
        print (feat+"_LOG", ret[feat])
    print ("SUM", sum(ret_linear.values()))
    print ("SUM_LOG", sum(ret.values()))
    return ret, ret_linear

if __name__ == '__main__':
    # Input two recs and one features file

    parser = argparse.ArgumentParser(description='Compute Commonality.')
    parser.add_argument('-rec', help='rec file')
    parser.add_argument('-feat', help='features file')
    parser.add_argument('-gamma', help='gamma')
    parser.add_argument('-users', help='reduced users file')

    args = parser.parse_args()
    gamma=0.95
    if args.gamma:
        gamma=float(args.gamma)
 
    if args.users:
        users = load_users(args.users)
    else:
        users = None
    total_items = 6040
    ret = commonality_log(args.rec, args.feat, total_items, users, gamma)
    #print (ret)

