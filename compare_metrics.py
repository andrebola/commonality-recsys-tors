import json
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator 

from utils import borda
from scipy.stats import kendalltau
from scipy.special import logsumexp

skip_files = {"input.run-only_female.txt": 1 , "input.run-non_eng_female.txt": 1}
neg_metrics = {'Prop_delta_diff', 'Prop_delta_abs', 'Prop_delta_sq', 'Prop_delta_kl', 'Exp_delta_diff', 'Exp_delta_abs', 'Exp_delta_sq', 'Exp_delta_kl'}

def metric_pairs(metric_vals, univ_metric_vals):
    vals_x = []
    vals_y = []
    names = []
    for run in metric_vals:
        if run not in univ_metric_vals:
            continue
        vals_x.append(metric_vals[run])
        vals_y.append(univ_metric_vals[run])
        names.append((run, univ_metric_vals[run], metric_vals[run]))
    return vals_x, vals_y, names

def plot_metric(agg_metrics, univ_metric, plot_metric):
    ret = {}
    for metric in agg_metrics:
        if metric == plot_metric:
            #print (metric)
            vals_x, vals_y, names = metric_pairs(agg_metrics[metric], univ_metric)
            #print (names)
            plt.plot([i[2] for i in names],[i[1] for i in  names], 'o', markersize=3)
            plt.grid()
            plt.ylabel('log(Commonality)', fontsize = 15)
            plt.xlabel('ndcg', fontsize = 15)
            #plt.ticklabel_format(style="sci")

            plt.yscale('symlog')
            ax = plt.gca()
            #ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            #ax.yaxis.set_minor_formatter(ScalarFormatter(useMathText=True))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            #ax.tick_params(axis='y', which='both', labelsize=14)
 
            plt.xticks(fontsize=14)
            #plt.yticks(fontsize=14)
            plt.subplots_adjust(bottom=0.15)
            plt.savefig('metrics.pdf', format='pdf', bbox_inches='tight')



def compare_all_metrics(agg_metrics, univ_metric):
    ret = {}
    for metric in agg_metrics:
        vals_x, vals_y, names = metric_pairs(agg_metrics[metric], univ_metric)
        c,p = kendalltau(vals_x, vals_y)
        ret[metric] = (c,p)
    return ret

def get_files_comm(folder):
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    metrics = {} 
    metrics_log = {}
    for f_name in onlyfiles:
        if f_name in skip_files:
            continue
        short_name = f_name.replace("input.", "")
        curr_log = {}
        with open(os.path.join(folder, f_name)) as metric_file:
            tsv_file = csv.reader(metric_file, delimiter=" ")
            for line in tsv_file:
                if "SUM" in line:
                    continue
                feat = line[0]
                score = line[1]

                if 'independent' in line[0]:
                    feat = line[0]+"-"+line[1]
                    score = line[2]
                
                if "_LOG" not in feat:
                    curr_log[feat] = float(score)


        metrics_log[short_name] = logsumexp(list(curr_log.values()), b=[1.0/len(curr_log)]*len(curr_log)) 
        #metrics_log[short_name] = np.median(list(curr_log.values()))
        metrics[short_name] = curr_log

    runs_compare = {}
    for m1 in metrics:
        runs_compare[m1] = 0
        for m2 in metrics:
            curr_comp = 0
            for feat in curr_log:
                if metrics[m1][feat] > metrics[m2][feat]:
                    curr_comp += 1
                #elif metrics[m2][feat] > metrics[m1][feat]:
                #    curr_comp -= 1
            runs_compare[m1] += (curr_comp*len(metrics))/float(len(curr_log))
    #runs_sum = {k: sum(v) for k,v in runs_compare.items()}
    runs_compare = {k: v for k, v in sorted(runs_compare.items(), key=lambda item: item[1], reverse=True)}
    #print (runs_compare)
    prefs = []
    for feat in curr_log:
        m1 = {k:v[feat] for k,v in metrics.items()}
        pref = {k: v for k, v in sorted(m1.items(), key=lambda item: item[1], reverse=True)}
        #print (pref)
        prefs.append(pref)
    bp = borda(prefs)
    #print (bp)
    #return  runs_compare, bp#metrics_log
    return bp, metrics_log

def get_files_fairness(folder):
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    metrics = {} 
    for f_name in onlyfiles:
        if f_name in skip_files:
            continue
        short_name = f_name.replace("input.", "")
        with open(os.path.join(folder, f_name)) as metric_file:
            tsv_file = csv.reader(metric_file, delimiter=" ")
            for line in tsv_file:
                metric = line[0]
                score = line[1]
                if metric not in metrics:
                    metrics[metric] = {}
                curr_score = float(score)
                if metric in neg_metrics:
                    curr_score *= -1
                metrics[metric][short_name] = curr_score
    return metrics

def get_files_div(folder):
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    metrics = {} 
    for f_name in onlyfiles:
        if f_name in skip_files:
            continue
        short_name = f_name.replace("input.", "")
        with open(os.path.join(folder, f_name)) as metric_file:
            tsv_file = csv.reader(metric_file, delimiter="\t")
            for line in tsv_file:
                metric = line[0]
                score = line[1]
                if metric not in metrics:
                    metrics[metric] = {}
                metrics[metric][short_name] = float(score)
    return metrics


def get_files_eval(folder):
    agg_metrics = {}
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    for f_name in onlyfiles:
        short_name = f_name.replace(".eval", "")
        with open(os.path.join(folder, f_name)) as rec_file:
            tsv_file = csv.reader(rec_file, delimiter="\t")
            curr_vals = {}
            for line in tsv_file:
                metric = line[0]
                user = line[1]
                perf = line[2]
                if metric not in curr_vals:
                    curr_vals[metric] = {}
                if user not in  curr_vals[metric]:
                     curr_vals[metric][user] = {}
                curr_vals[metric][user] = float(perf)

            for metric in curr_vals:
                if metric not in agg_metrics:
                    agg_metrics[metric] = {}
                if short_name not in agg_metrics[metric]:
                    agg_metrics[metric][short_name] = {}
                agg_metrics[metric][short_name] = sum(curr_vals[metric].values()) / float(len(curr_vals[metric]))
    return agg_metrics

if __name__ == '__main__':
    # Get each system's performance according to different metrics
    # Compute comparison according to a metrics

    folder = "../../scratch/results/trec_eval/ml-1m/ml-1m/"
    metrics = get_files_eval(folder)
    folder_comm = "/network/projects/_groups/musai/results/comm-gamma5-ml-1m-multi/"
    comm_metric, comm_metric_log = get_files_comm(folder_comm)
    folder_fairness = "/network/projects/_groups/musai/results/fairness-ml-1m/"
    fairness_metric = get_files_fairness(folder_fairness)
    metrics.update(fairness_metric)

    folder_div = "/network/projects/_groups/musai/results/diversity-ml-1m/"
    div_metric = get_files_div(folder_div)
    metrics.update(div_metric)

    #for k,v in metrics.items():
    #    print (k,v)

    plot_metric(metrics, comm_metric, "ndcg")
    metrics_comp = compare_all_metrics(metrics, comm_metric)
    for k,v in metrics_comp.items():
        print ("BORDA",",", k, ",", v[0], ",", v[1])
    #print ("LOG")
    metrics_comp = compare_all_metrics(metrics, comm_metric_log)
    for k,v in metrics_comp.items():
        print ("MEAN", ",", k, ",", v[0], ",", v[1])
