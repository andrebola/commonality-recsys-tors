import csv
import json

def get_prod(values):
    curr_prods = 1.0
    for i in values:
        curr_prods *= i
    return curr_prods

def load_features(features_file):
    ret = {}
    with open(features_file) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            if line[1] not in ret:
                ret[line[1]] = []
            ret[line[1]].append(line[0])
    return ret

def load_recs(recs_file):
    ret = {}
    with open(recs_file) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            if line[0] not in ret:
                ret[line[0]] = []
            ret[line[0]].append(line[1])
            # TODO: Sort based on score
    return ret

def load_recs_trec_format(recs_file):
    ret = {}
    with open(recs_file) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            if line[0] not in ret:
                ret[line[0]] = []
            ret[line[0]].append(line[2])
            # TODO: Sort based on score
    return ret

def load_users(users_file):
    ret = {}
    with open(users_file) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            ret[line[0]] = 1
    return ret

def load_rels(rels_file, min_val=0):
    ret = {}
    with open(rels_file) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            if line[0] not in ret:
                ret[line[0]] = {}
            if int(line[3]) > min_val:
                ret[line[0]][line[2]] = int(line[3])
    return ret

def borda(prefs):
    cnts = {}
    n = None
    for pref in prefs:
        if n is None:
            n = len(pref)
        for idx, key in enumerate(pref.keys()):
            if not key in cnts:
                cnts[key] = 0
            cnts[key] += (n-idx)
    return {k: v for k, v in sorted(cnts.items(), key=lambda item: item[1], reverse=True)}


