RECS="../../recall/data/runs/ml-1m/ml-1m_m18/"
FEAT="../../elliot/data/movielens_1m/ml-features.tsv"
REL="../../recall/data/qrels/ml-1m/full_ml-1m.txt"
TRAIN="../../elliot/data/movielens_1m/train.txt"
REDUCED="/network/projects/_groups/musai/data/ml-1m/reduced/users/"
RESULTS="/network/projects/_groups/musai/results/div-ml-1m-user-red"

cat $REL | awk -F"\t" '{print $1 "\t" $3 "\t" $4 ;}'  > /tmp/gt.txt
for k in 1 2 3; do \
    for i in 10 20 30 40 50 60 70 80 90 ; do \
        mkdir -p $RESULTS/reduced_$k/$i/
        python reduce_users.py /tmp/gt.txt ${REDUCED}${k}_${i}.txt /tmp/red_gt.txt
        for f in $RECS*.gz ; do 
          FNAME=$(basename "$f" .gz)
          zcat "$f" | awk -F"\t" '{print $1 "\t" $3 "\t" $5 ;}'  > /tmp/$FNAME
	  python reduce_users.py /tmp/$FNAME ${REDUCED}${k}_${i}.txt /tmp/red_rec.txt
	  python reduce_users.py  $TRAIN ${REDUCED}${k}_${i}.txt /tmp/red_train.txt 
          java -cp "../target/community-1.jar:../target/dependency/*" com.mila.recsys.MetricExample /tmp/red_train.txt /tmp/red_gt.txt $FEAT /tmp/red_rec.txt 4 > $RESULTS/reduced_$k/$i/$FNAME;
        done
   done
done



