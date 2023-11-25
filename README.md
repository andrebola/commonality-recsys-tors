## Commonality

This repository contains the code to reproduce the results of the article entitled "Measuring Commonality in Recommendation of Cultural Content: Recommender Systems to Enhance Cultural Citizenship", by Andres Ferraro, Gustavo Ferreira, Fernando Diaz, and Georgina Born.

Requirements:
 - Python 3.6.10
 - Dependencies are listed in requirements.txt

The experiments that we conduct in the work require 3 datasets. To obtain these datasets you can access their respective websites:

 - [Movielens-1m](https://grouplens.org/datasets/movielens/1m/)
 - [Last.fm-2b](http://www.cp.jku.at/datasets/LFM-2b/)
 - [Librarything](https://cseweb.ucsd.edu/~jmcauley/datasets.html#social_data): 

The experiments are based on recommendations produced by multiple algorithms and metrics computed over these recommendations. To compute these metrics we rely on standard implementations that are widely used by the community:
 - To compute diversity metrics we use the library [RankSys](https://github.com/RankSys/RankSys)
 - To compute accuracy metrics we use [trec_eval](https://github.com/usnistgov/trec_eval)
 - To compute the recommendations for the Last-fm dataset we use [Elliot](https://github.com/sisinflab/elliot)
 - Recommendations for Librarything and ML-1m were provided by Valcarce et al. in "On the robustness and discriminative power of information retrieval metrics for top-N recommendation"

Once all the metrics mentioned above are computed, the following steps can be applied:
 1) Compute Commonality and Fairness metrics: `compute_comm.sh`
 2) Compute correlation between the metrics (Section 5.3.3): `plot_compare_comm.py`
 3) Compute metrics with missing category labels (Section 5.3.4) `compute_red_feats.sh` and `compute_red_all_feats.sh`
 4) Compute metrics with reduced users (Section 5.3.4) `compute_comm_reduce.sh` and `compute_red_div.sh`

## Cite

Andres Ferraro, Gustavo Ferreira, Fernando Diaz, Georgina Born "Measuring Commonality in Recommendation of Cultural Content: Recommender Systems to Enhance Cultural Citizenship". ACM Transactions on Recommender Systems. 2022

