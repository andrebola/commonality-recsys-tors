import json
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator 

from scipy.special import logsumexp

mapped_names = {'sa': 'south-africa', 'female': 'female', 'na': 'north-africa', 'me': 'middle-east', 'ea': 'east-africa', 'ma': 'middle-africa', 'sea': 'southeast-asia', 'wa': 'west-arica', 'ca': 'central-asia', 'non-binary': 'non-binary'}

skip_files = {"input.run-only_female.txt": 1 , "input.run-non_eng_female.txt": 1}

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def get_files_comm(folder, runs):
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    metrics = {} 
    metrics_log = {} 
    for f_name in onlyfiles:
        if f_name in skip_files:
            continue
        short_name = f_name.replace("input.", "")
        if short_name not in runs:
            continue
        if short_name not in metrics_log:
            metrics_log[short_name] = {}
            metrics[short_name] = {}
        with open(os.path.join(folder, f_name)) as metric_file:
            tsv_file = csv.reader(metric_file, delimiter=" ")
            for line in tsv_file:
                if "SUM" in line[0]:
                    continue
                feat = line[0]
                score = line[1]
                if 'independent' in line[0]:
                    feat = line[0]+"-"+line[1]
                    score = line[2]
                if "LOG" in feat: 
                    metrics_log[short_name][feat] = float(score)
                else:
                    metrics[short_name][feat] = float(score)
    return metrics, metrics_log


def plot_compare(comm_metric, label_map):
    barWidth = 1./(1+len(comm_metric))
    fig = plt.subplots(figsize =(12, 8))

    cmap = get_cmap(len(comm_metric)+1, "Accent")
    # Set position of bar on X axis
    br = []
    last_width = []
    for label,vals in comm_metric.items():
        if len(last_width) == 0:
            br.append(np.arange(len(vals)))
        else:
            br.append(np.arange(len(vals)))
            #br.append([x + barWidth for x in last_width])
        last_width = br[-1]

    # Make the plot
    all_max = []
    all_min = []
    for i, (label,vals) in enumerate(comm_metric.items()):
        for k,j in enumerate(vals.values()):
            if k >= len(all_max):
                all_max.append(j)
                all_min.append(j)
            if j > all_max[k]:
                all_max[k] = j
            if j < all_min[k]:
                all_min[k] = j
    all_vals = []
    for k,i in enumerate(all_min):
        all_vals.append(i)
        all_vals.append(all_max[k])
    from matplotlib import collections as matcoll
    linecoll = matcoll.LineCollection([[(br[0][k], i),(br[0][k],j)] for k, (i,j) in enumerate(zip(all_max,all_min))], colors='dimgray', linestyles='dashed')

    for i, (label,vals) in enumerate(comm_metric.items()):
        a = np.array(list(vals.values()), np.float128)
        b = a.max()
        l = b + np.log((np.exp(a-b)).sum()) - np.log(len(a))
        
        #l= np.median(list(vals.values()))
        #l = logsumexp(list(vals.values()), b=[1.0/len(vals)]*len(vals))
        print (label)
        print (vals)
        print (l)
        plt.plot(br[i], vals.values(), color =cmap(i), alpha=.7,marker='o',linestyle='', markersize=10, label =label_map[label])
        plt.axhline(y=l, color=cmap(i), linestyle='-')

    # Adding Xticks
    plt.xlabel('Category', fontsize = 15)
    plt.ylabel('log(Commonality)', fontsize = 15)
    plt.yscale('symlog')
    ax = plt.gca()
    ax.add_collection(linecoll, )
    #ax.set_ylim([-5e5, -7e4])
    ax.set_ylim([-1.6e6, -2.5e5])
    start, end = ax.get_ylim()
    print (start, end)
    ax.yaxis.set_ticks(np.arange(start, end,100000))
    #ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    #ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax.yaxis.set_minor_formatter(ScalarFormatter(useMathText=True))
    #ax.tick_params(axis='y', which='minor', labelsize=15)
    plt.yticks(fontsize=13)
    #ax.yaxis.set_major_formatter(ScalarFormatter())
    #ax.minorticks_off()
    plt.xticks(br[-1], [mapped_names[i] for i in vals.keys()], rotation=45, fontsize=13)
    plt.legend(loc="upper right", prop={'size': 13})
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('compare-tors-lf.pdf', format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    # Get each system's performance according to different metrics
    # Compute comparison according to a metrics

    runs = {"run-Random-ml-1M-fold1.txt": "Random", "run-BPRMF-ml-1M-fold1.txt": "BPRMF", "run-Pop-ml-1M-fold1.txt":"Pop"}
    #runs = {"run-RM2-ml-1M-fold1.txt": "RM2", "run-RSV-ml-1M-fold1.txt": "RSV", "run-RW-ml-1M-fold1.txt": "RW", 
    #        "run-SLIM-ml-1M-fold1.txt": "SLIM", "run-SVD-ml-1M-fold1.txt": "SVD", "run-UIR-ml-1M-fold1.txt": "UIR"}
    runs = {"Random.txt": "Random", "BPRMF.txt": "BPRMF", "MostPop.txt":"Pop"}

    #run-NNCosNgbr-UB-ml-1M-fold1.txt
    folder_comm = "/network/projects/_groups/musai/results/comm-gamma5-ml-1m-multi/"
    folder_comm = "/network/projects/_groups/musai/results/comm-gamma5-lf-2b/"
    comm_metric, comm_metric_log = get_files_comm(folder_comm, runs.keys())

    #plot_compare(comm_metric_log, runs)
    plot_compare(comm_metric, runs)
    
