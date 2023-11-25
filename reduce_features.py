import random
import argparse
from utils import load_features

def print_feats(features):

    for feat, items in features.items():
        for i in items:
            print (i+"\t"+feat)

def reduce_cat(features_file, perc):
    features = load_features(features_file)

    max_cat = 0
    top_category = None
    for c, c_items in features.items():
        if max_cat < len(c_items):
            max_cat = len(c_items)
            top_category = c

    red_items = random.sample(features[top_category], int(perc*max_cat))
    features[top_category] = red_items

    print_feats(features)

def reduce_all_cat(features_file, perc):
    features = load_features(features_file)
    min_cat = 9999
    for c, c_items in features.items():
        if min_cat > len(c_items):
            min_cat = len(c_items)

    for c, c_items in features.items():
        len_c = len(c_items)
        new_len = (len_c - min_cat)*perc + min_cat
        new_len = max(10, int(new_len))
        red_items = random.sample(c_items, min(new_len, len_c))
        features[c] = red_items
    
    print_feats(features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reduce items with features.')
    parser.add_argument('-f', help='file')
    parser.add_argument('-p', help='percentage')
    parser.add_argument('-t', help='type')

    args = parser.parse_args()
    perc= (100 - int(args.p)) / 100

    if args.t == "ALL":
        reduce_all_cat(args.f, perc)
    else:
        reduce_cat(args.f, perc)
