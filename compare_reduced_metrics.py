import json
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from utils import borda
from scipy.special import logsumexp
from scipy.stats import kendalltau
from compare_metrics import get_files_fairness, get_files_comm, get_files_eval, get_files_div


neg_metrics = {'Prop_delta_diff', 'Prop_delta_abs', 'Prop_delta_sq', 'Prop_delta_kl', 'Exp_delta_diff', 'Exp_delta_abs', 'Exp_delta_sq', 'Exp_delta_kl'}
"""
def compare_metrics(agg_metrics, red_metrics):
    ret = {}
    for metric in red_metrics:
        for red in red_metrics[metric]:
            x_list = []
            y_list = []
            for sysname in red_metrics[metric][red]:
                if sysname not in metrics[metric]:
                    continue
                x_list.append(red_metrics[metric][red][sysname])
                y_list.append(metrics[metric][sysname])
            c,p = kendalltau(x_list, y_list)
            ret[metric+"_"+str(red)] = (c,p)
    return ret
"""

def compare_metrics(metrics, red_metrics):
    ret = {}
    for it in [1,2,3]:
        for red in sorted(red_metrics.keys()):
            x_list = []
            y_list = []
            for sysname in red_metrics[red]:
                if sysname not in metrics:
                    continue
                x_list.append(red_metrics[red][sysname][it-1])
                y_list.append(metrics[sysname])
            c,p = kendalltau(x_list, y_list)
            #ret[str(red)] = (c,p)
            if str(red) not in ret:
                ret[str(red)] = []
            ret[str(red)].append(c)
            #print (x_list, y_list, c,p, red)
    for red in ret:
        ret[red] = (np.mean(ret[red]), np.std(ret[red]))
    return ret

def get_files_comm_reduced(folder, red_folder="reduced_"):
    metrics = {} 
    for it in [1, 2, 3]:
        for red in [10,20,30,40,50,60,70,80,90]:
            curr_location = os.path.join(folder,red_folder+str(it), str(red))
            onlyfiles = [f for f in os.listdir(curr_location) if os.path.isfile(os.path.join(curr_location, f))]
            for f_name in onlyfiles:
                short_name = f_name.replace("input.", "")
                with open(os.path.join(curr_location, f_name)) as metric_file:
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
                            if red not in metrics:
                                metrics[red] = {}
                            if short_name  not in metrics[red]:
                                metrics[red][short_name] = {}
                            if feat not in metrics[red][short_name]:
                                metrics[red][short_name][feat] = []
                            metrics[red][short_name][feat].append(float(score))
    
    bp_metrics = {}
    mean_metrics = {}
    for red in metrics:
        mean_metrics[red] = {}
        for i in metrics[red]:
            #for feat in metrics[red][i]:
            #    metrics[red][i][feat] = np.mean(metrics[red][i][feat])
            for it in [1,2,3]:
                if i not in mean_metrics[red]:
                    mean_metrics[red][i] = []
                vals = [j[it-1] for j in metrics[red][i].values()]
                mean_metrics[red][i].append(logsumexp(vals, b=[1.0/len(metrics[red][i])]*len(metrics[red][i])))
                #MEDIAN?

        for it in [1,2,3]:
            prefs = []
            for feat in metrics[red][i]:
                m1 = {k:v[feat][it-1] for k,v in metrics[red].items()}
                pref = {k: v for k, v in sorted(m1.items(), key=lambda item: item[1], reverse=True)}
                prefs.append(pref)
            ret_borda = borda(prefs)
            for s in ret_borda:
                if red not in bp_metrics:
                    bp_metrics[red]={}
                if s not in bp_metrics[red]:
                    bp_metrics[red][s] = []
                bp_metrics[red][s].append(ret_borda[s])

    return bp_metrics, mean_metrics


def get_files_fairness_reduced(folder, red_folder="reduced_", delimiter=" "):
    metrics = {} 
    for it in [1, 2, 3]:
        for red in [10,20,30,40,50,60,70,80,90]:
            curr_location = os.path.join(folder,red_folder+str(it), str(red))
            onlyfiles = [f for f in os.listdir(curr_location) if os.path.isfile(os.path.join(curr_location, f))]
            for f_name in onlyfiles:
                short_name = f_name.replace("input.", "")
                with open(os.path.join(curr_location, f_name)) as metric_file:
                    tsv_file = csv.reader(metric_file, delimiter=delimiter)
                    for line in tsv_file:
                        metric = line[0]
                        score = line[1]
                        if metric not in metrics:
                            metrics[metric] = {}
                        if red not in metrics[metric]:
                            metrics[metric][red] = {}
                        if short_name not in metrics[metric][red]:
                            metrics[metric][red][short_name] =[]
                        curr_score = float(score)
                        if metric in neg_metrics:
                            curr_score *= -1
                        metrics[metric][red][short_name].append(curr_score)
 
    #for m in metrics:
    #    for i in metrics[metric]:
    #        for red in metrics[metric][i]:
    #            metrics[m][i][red] = np.mean(metrics[m][i][red])
    return metrics


def get_files_eval_reduced(folder):
    agg_metrics = {} 
    for it in [1,2]:
        curr_location = os.path.join(folder,'reduced_'+str(it))
        onlyfiles = [f for f in os.listdir(curr_location) if os.path.isfile(os.path.join(curr_location, f))]
        for f_name in onlyfiles:
            red = f_name.split(".txt_")[1].replace(".eval", "")
            short_name = f_name.replace("_"+red+".eval", "")
            with open(os.path.join(curr_location, f_name)) as rec_file:
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
                    if red not in agg_metrics[metric]:
                        agg_metrics[metric][red] = {}
                    if short_name not in agg_metrics[metric][red]:
                        agg_metrics[metric][red][short_name] = []
                    agg_metrics[metric][red][short_name].append(sum(curr_vals[metric].values()) / float(len(curr_vals[metric])))
    for m in agg_metrics:
        for i in agg_metrics[m]:
            for red in agg_metrics[m][i]:
                agg_metrics[m][i][red] = np.mean(agg_metrics[m][i][red])
    return agg_metrics


if __name__ == '__main__':
    # Get each system's performance according to different metrics
    # Compute comparison according to a metrics

    # Diversity:
    folder_div = "/network/projects/_groups/musai/results/div-ml-1m-user-red/"
    red_div_metric = get_files_fairness_reduced(folder_div, delimiter="\t")
    folder_div = "/network/projects/_groups/musai/results/diversity-ml-1m/"
    div_metric = get_files_div(folder_div)

    for metric in red_div_metric.keys():
        metrics_comp = compare_metrics(div_metric[metric], red_div_metric[metric])
        for k,v in metrics_comp.items():
            print (metric, ",", k, ",", v[0], ",", v[1])  

    # Commonality:
    folder_univ = "/network/projects/_groups/musai/results/comm-gamma5-ml-1m-user-red/"
    red_comm_metric_borda, red_comm_metric_mean = get_files_comm_reduced(folder_univ)

    folder_univ = "/network/projects/_groups/musai/results/comm-gamma5-ml-1m/"
    comm_metric_borda, comm_metric_mean = get_files_comm(folder_univ)

    metrics_comp = compare_metrics(comm_metric_mean, red_comm_metric_mean)
    for k,v in metrics_comp.items():
        print ("MEAN",",",k,",",v[0],",", v[1])

    metrics_comp = compare_metrics(comm_metric_borda, red_comm_metric_borda)
    for k,v in metrics_comp.items():
        print ("BORDA",",",k,",",v[0],",", v[1])
    

    # Fairness:
    folder_div = "/network/projects/_groups/musai/results/fairness-ml-1m-user-red/"
    red_fairness_metric = get_files_fairness_reduced(folder_div)
    folder_fairness = "/network/projects/_groups/musai/results/fairness-ml-1m/"
    fairness_metric = get_files_fairness(folder_fairness)

    for metric in red_fairness_metric.keys():
        metrics_comp = compare_metrics(fairness_metric[metric], red_fairness_metric[metric])
        for k,v in metrics_comp.items():
            print (metric,",", k, ",", v[0],",", v[1])  

