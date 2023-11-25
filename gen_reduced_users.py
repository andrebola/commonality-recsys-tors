import os
import csv
import argparse
import random
from utils import load_rels

def save_output(output_file, reduction, rel):
    with open(output_file+"_"+reduction+".txt", mode='w') as w_file:
        w_writer = csv.writer(w_file, delimiter='\t')
        for u in rel:
            w_writer.writerow([u])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Commonality.')
    parser.add_argument('-rel', help='relevance file')
    parser.add_argument('-o', help='output file')

    args = parser.parse_args()
 
    rels = load_rels(args.rel)
    users = list(rels.keys())
    for i in range(10,100, 10):
        n = len(users)
        perc= (100 - i) / 100
        curr_sample = random.sample(users, int(n*perc))
        save_output(args.o, str(i), curr_sample)

