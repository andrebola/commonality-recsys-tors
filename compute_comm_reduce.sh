RECS="../../recall/data/runs/ml-1m/ml-1m_m18/"
FEAT="../../elliot/data/movielens_1m/ml-features.tsv"
REL="../../recall/data/qrels/ml-1m/full_ml-1m.txt"
TRAIN="../../elliot/data/movielens_1m/train.txt"
REDUCED="/network/projects/_groups/musai/data/ml-1m/reduced/users/"
RESULTS="/network/projects/_groups/musai/results/comm-gamma5-ml-1m-user-red/"
RESULTS_FAIR="/network/projects/_groups/musai/results/fairness-ml-1m-user-red/"

#cat ../recall/data/qrels/ml-1m/full_ml-1m.txt | awk -F"\t" '{print $1 "\t" $3 "\t" $4 ;}'  > /tmp/gt.txt
for k in 1 2 3; do \
    for i in 10 20 30 40 50 60 70 80 90 ; do \
        mkdir -p $RESULTS/reduced_$k/$i/
        mkdir -p $RESULTS_FAIR/reduced_$k/$i/
        for f in $RECS*.gz ; do 
          FNAME=$(basename "$f" .txt.gz)
          if [[ ! -f $RESULTS/reduced_$k/$i/$FNAME.txt ]]
          then
          zcat "$f" | awk -F"\t" '{print $1 "\t" $3 "\t" $5 ;}'  > /tmp/$FNAME
          python -u commonality.py -rec /tmp/$FNAME -feat $FEAT -gamma "0.5" -users ${REDUCED}${k}_${i}.txt > $RESULTS/reduced_$k/$i/$FNAME.txt
          python -u fairness.py -rec /tmp/$FNAME -feat $FEAT -rel $REL -users ${REDUCED}${k}_${i}.txt -train $TRAIN > $RESULTS_FAIR/reduced_$k/$i/$FNAME.txt
          fi
        done
   done
done



