RECS="../../recall/data/runs/ml-1m/ml-1m_m18/"
FEAT="../../elliot/data/movielens_1m/ml-features.tsv"
REL="../../recall/data/qrels/ml-1m/full_ml-1m.txt"
TRAIN="../../elliot/data/movielens_1m/train.txt"
RESULTS="/network/projects/_groups/musai/results/comm-gamma5-ml-1m-multi/"
RESULTS_FAIR="/network/projects/_groups/musai/results/fairness-ml-1m/"

#cat ../recall/data/qrels/ml-1m/full_ml-1m.txt | awk -F"\t" '{print $1 "\t" $3 "\t" $4 ;}'  > /tmp/gt.txt

for f in $RECS*.gz ; do 
    if [[ ! -f $RESULTS$(basename "$f" .txt.gz).txt ]]
    then
    zcat "$f" | awk -F"\t" '{print $1 "\t" $3 "\t" $5 ;}'  > /tmp/$(basename "$f" .gz)
#java -cp "target/community-1.jar:target/dependency/*" com.mila.recsys.MetricExampleInd /tmp/gt.txt ../elliot/data/movielens_1m/features.txt /tmp/file1.txt > ./results/ml-1m/$(basename "$f" .gz);
    python -u commonality.py -rec /tmp/$(basename "$f" .gz) -feat $FEAT -gamma "0.5" > $RESULTS$(basename "$f" .txt.gz).txt
    #python -u fairness.py -rec /tmp/$(basename "$f" .gz) -feat $FEAT -rel $REL -train $TRAIN > $RESULTS_FAIR$(basename "$f" .txt.gz).txt
    #rm /tmp/file1.txt
    fi
done

