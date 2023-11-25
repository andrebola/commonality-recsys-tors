RECS="../../recall/data/runs/ml-1m/ml-1m_m18/"
FEAT="../../elliot/data/movielens_1m/ml-features.tsv"
REL="../../recall/data/qrels/ml-1m/full_ml-1m.txt"
TRAIN="../../elliot/data/movielens_1m/train.txt"
REDUCED="/network/projects/_groups/musai/data/ml-1m/reduced/all_features/"
RESULTS="/network/projects/_groups/musai/results/comm-gamma5-ml-1m-feat-red"
RESULTS_FAIR="/network/projects/_groups/musai/results/fairness-ml-1m-feat-red"
RESULTS_DIV="/network/projects/_groups/musai/results/diversity-ml-1m-feat-red"

cat $REL | awk -F"\t" '{print $1 "\t" $3 "\t" $4 ;}'  > /tmp/gt.txt

for k in 4 5; do \
    for i in 10 30 50 70 90 100; do \
    FEAT_K=${REDUCED}${k}_${i}.txt
    if [[ ! -f $FEAT_K ]]
    then
        python reduce_features.py -f $FEAT -p $i -t "ALL" > $FEAT_K
    fi
    #mkdir -p $RESULTS/all_reduced_$k/$i/
    #mkdir -p $RESULTS_FAIR/all_reduced_$k/$i/
    #mkdir -p $RESULTS_DIV/all_reduced_$k/$i/
 
    for f in $RECS*.gz ; do 
        FNAME=$(basename "$f" .txt.gz)
        #if [[ ! -f $RESULTS/all_reduced_$k/$i/$FNAME.txt ]]
        #then
        zcat "$f" | awk -F"\t" '{print $1 "\t" $3 "\t" $5 ;}'  > /tmp/$FNAME
        java -cp "../target/community-1.jar:../target/dependency/*" com.mila.recsys.MetricExample $TRAIN /tmp/gt.txt $FEAT_K /tmp/$FNAME 4 > $RESULTS_DIV/all_reduced_$k/$i/$FNAME.txt;
        #python -u commonality.py -rec /tmp/$FNAME -feat $FEAT_K -gamma "0.5" > $RESULTS/all_reduced_$k/$i/$FNAME.txt
        #python -u fairness.py -rec /tmp/$FNAME -feat $FEAT_K -rel $REL -train $TRAIN > $RESULTS_FAIR/all_reduced_$k/$i/$FNAME.txt
        #rm /tmp/file1.txt
        #fi
        done
    done
done
