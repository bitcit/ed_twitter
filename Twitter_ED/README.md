## Steps to reproduce dataset generation:
Generate twitter data by following these steps and copy all files to this directory:
https://github.com/LejlaMetohajrova/deep-ed-pytorch#generating-twitter-data

### Take only examples wih GT values
```
grep -v -P 'GT:\t-1' Brian_Collection.csv > brian_gt.csv
grep -v -P 'GT:\t-1' Mena_Collection.csv > mena_gt.csv
grep -v -P 'GT:\t-1' Microposts2014_train.csv > micro_gt.csv
```

#mentions | dataset
-----------------------|--------
3147 | micro_gt.csv
1006 | brian_gt.csv
423 | mena_gt.csv

### Shuffle examples
```
shuf brian_gt.csv > brian_gt_shuf.csv
shuf mena_gt.csv > mena_gt_shuf.csv
shuf micro_gt.csv > micro_gt_shuf.csv
```

### Train-val-test split
```
mkdir -p train
head -n 2517 micro_gt_shuf.csv > train/micro_train.csv
head -n 806 brian_gt_shuf.csv > train/brian_train.csv
head -n 339 mena_gt_shuf.csv > train/mena_train.csv
```
```
mkdir -p val
mkdir -p test
tail -n 630 micro_gt_shuf.csv | head -n 315 > val/micro_val.csv
tail -n 630 micro_gt_shuf.csv | tail -n 315 > test/micro_test.csv
tail -n 200 brian_gt_shuf.csv | head -n 100 > val/brian_val.csv
tail -n 200 brian_gt_shuf.csv | tail -n 100 > test/brian_test.csv
tail -n 84 mena_gt_shuf.csv | head -n 42 > val/mena_val.csv
tail -n 84 mena_gt_shuf.csv | tail -n 42 > test/mena_test.csv
```

Dataset | Train (80%) | Val (10%) | Test (10%)
--------|-------------|-----------|-----------
micro | 2517 | 315 | 315
brian | 806 | 100 | 100
mena | 339 | 42 | 42
total | 3662 | 457 | 457

### Generate associated conll files
```
python3 gen_train_val_test.py -conll_file Microposts2014_train.conll \
    -csv_file train/micro_train.csv
python3 gen_train_val_test.py -conll_file Microposts2014_train.conll \
    -csv_file val/micro_val.csv
python3 gen_train_val_test.py -conll_file Microposts2014_train.conll \
    -csv_file test/micro_test.csv
```

```
python3 gen_train_val_test.py -conll_file Brian_Collection.conll \
    -csv_file train/brian_train.csv
python3 gen_train_val_test.py -conll_file Brian_Collection.conll \
    -csv_file val/brian_val.csv
python3 gen_train_val_test.py -conll_file Brian_Collection.conll \
    -csv_file test/brian_test.csv
```

```
python3 gen_train_val_test.py -conll_file Mena_Collection.conll \
    -csv_file train/mena_train.csv
python3 gen_train_val_test.py -conll_file Mena_Collection.conll \
    -csv_file val/mena_val.csv
python3 gen_train_val_test.py -conll_file Mena_Collection.conll \
    -csv_file test/mena_test.csv
```

The number of tweets per dataset:

Dataset | Train | Val | Test
--------|-------|-----|-----
micro | 1331 | 283 | 287
brian | 575 | 98 | 95
mena | 152 | 37 | 39
total | 2058 | 418 | 421

Concatenate datasets:
```
cat brian_train.conll mena_train.conll micro_train.conll > twitter_train.conll
cat brian_test.conll mena_test.conll micro_test.conll > twitter_test.conll
cat brian_val.conll mena_val.conll micro_val.conll > twitter_val.conll
```

Original twitter data come from: \
[1] Locke, B. and Martin, J. (2009). Named entity recognition: Adapting to microblogging. Senior Thesis, University of Colorado. \
[2] Habib, M. B. and van Keulen, M. (2012). Unsupervised improvement of named entity extraction in short informal context using disambiguation clues. In Proceedings of the Workshop on Semantic Web and Information Extraction (SWAIE 2012), pages 1–10. \
[3] A. E. Cano Basave, G. Rizzo, A. Varga, M. Rowe, M. Stankovic, and A.-S. Dadzie (2014). Making Sense of Microposts (#Microposts2014) Named Entity Extraction & Linking Challenge. In Proceedings of #Microposts2014, pages 54–60, 2014.
