# Training entity embeddings and data preprocessing for entity disambiguation
Data preprocessing and training entity embeeddings inspired by: \
Ganea, O. E., & Hofmann, T. (2017). Deep joint entity disambiguation with local neural attention. arXiv preprint arXiv:1704.04920.

## How to run the code and reproduce the results

### Install dependencies
Install [Pytorch](https://pytorch.org/get-started/locally/), [nltk](https://www.nltk.org/install.html), [gensim](https://radimrehurek.com/gensim/install.html).

E.g.:

```
pip3 install torch torchvision
pip3 install nltk
pip3 install gensim
pip3 install unidecode
```  

### Run everything at once with this command
```
./run_all.sh
```

### Or follow the steps below
#### Data preprocessing

1) Create $DATA_PATH (e.g. export DATA_PATH=data/) directory and subdirectory $DATA_PATH/generated/ that will contain all files generated in the next steps.

2) Download data files needed for training and testing from [this link](https://drive.google.com/uc?id=0Bx8d3azIm_ZcbHMtVmRVc1o5TWM&export=download).
 Download basic_data.zip, unzip it and place the basic_data directory in $DATA_PATH/. All generated files will be build based on files in this basic_data/ directory.

3) Download pre-trained Word2Vec vectors GoogleNews-vectors-negative300.bin.gz from https://code.google.com/archive/p/word2vec/.
Unzip it and place the bin file in the folder $DATA_PATH/basic_data/wordEmbeddings/Word2Vec.

4) Generate entity-mention prior from wikipedia:

```
python3 -m data_gen.gen_p_e_m.gen_p_e_m_from_wiki -root_data_dir $DATA_PATH
```
 
5) Merge wikipedia_p_e_m.txt and crosswikis_p_e_m.txt : 

```
python3 -m data_gen.gen_p_e_m.merge_crosswikis_wiki -root_data_dir $DATA_PATH
```

6) Create yago_p_e_m.txt: 

```
python3 -m data_gen.gen_p_e_m.gen_p_e_m_from_yago -root_data_dir $DATA_PATH
```

7) Create a file ent_wiki_freq.txt with entity frequencies: 

```
python3 -m entities.ent_name2id_freq.e_freq_gen -root_data_dir $DATA_PATH
```

8) Generate all entity disambiguation datasets in a CSV format needed in our training stage: 

```
mkdir $DATA_PATH/generated/test_train_data/
python3 -m data_gen.gen_test_train_data.gen_all -root_data_dir $DATA_PATH
```

9) Verify stats of these files as explained in the header comments of files `gen_ace_msnbc_aquaint_csv.py` and `gen_aida_test.py`.

10) Generate training data from Wikipedia:

    i) From Wikipedia canonical pages:
    ```
    python3 -m data_gen.gen_wiki_data.gen_ent_wiki_w_repr -root_data_dir $DATA_PATH
    ```
    
    ii) From context windows surrounding Wiki hyperlinks:
    ```
    python3 -m data_gen.gen_wiki_data.gen_wiki_hyp_train_data -root_data_dir $DATA_PATH
    ```

11) Compute unigram counts from Wikipedia corpus:

```
python3 -m words.w_freq.w_freq_gen -root_data_dir $DATA_PATH
```

12) Compute the restricted training data for learning entity embeddings by using only candidate entities from the relatedness datasets and all ED sets:
    
    i) From Wikipedia canonical pages:
    ```
    python3 -m entities.relatedness.filter_wiki_canonical_words_RLTD -root_data_dir $DATA_PATH
    ```
    
    ii) From context windows surrounding Wiki hyperlinks:
    ```
    python3 -m entities.relatedness.filter_wiki_hyperlink_contexts_RLTD -root_data_dir $DATA_PATH
    ```

#### Training entity embeddings

```
mkdir $DATA_PATH/generated/ent_vecs
python3 -u -m entities.learn_e2v.learn_a -root_data_dir $DATA_PATH |& tee log_train_entity_vecs
```

### Generating TwitterNEED Data

1) Download data from https://github.com/badiehm/TwitterNEED

2) Place TwitterNEED directory under `$DATA_PATH/basic_data/` directory

3) Generate test-train csv files for each twitter xml:
 ```
 python3 -m data_gen.gen_test_train_data.gen_from_tweets -root_data_dir $DATA_PATH \
    -twitter_input basic_data/TwitterNEED/twitter_file.xml \
    -twitter_in_format xml \
    -out_file generated/test_train_data/twitter_file.csv 
 ```
 
 4) Generate conll files for each twitter xml:
 ```
 python3 -m data_gen.conll_from_tweets -root_data_dir $DATA_PATH \
    -twitter_input basic_data/TwitterNEED/twitter_file.xml \
    -twitter_in_format xml \
    -out_file generated/twitter_file.conll
 ```
 
### Converting Microposts2016
1) Download Microposts2016 data with tweets (e.g. from [here](https://drive.google.com/drive/folders/0B7oVKINifZRDejVmdlh6dWdBZUU?zx=wzlnhh617woh)) and rename the folders to `dev`, `test` and `training`.
2) Place the directory with Microposts 2016 data under `$DATA_PATH/basic_data/`.

Example input `file.tsv`:
```
675572272418021377	RT @VisitAbuDhabi: You could win 3 days #InAbuDhabi with us & Etihad Airways, VIP #StarWars Premiere tickets & more! Join now: https://t.co…
```

Example input `file.gs`:
```
675572272418021377	62	76	http://dbpedia.org/resource/Etihad_Airways	1	Organization
675572272418021377	83	91	http://dbpedia.org/resource/Star_Wars	1	Product
675572272418021377	41	51	http://dbpedia.org/resource/Abu_Dhabi	1	Location
675572272418021377	4	17	NIL17	1	Organization
```
 
3) Generate test-train csv files for each of the splits folders (example for training split):
```
 python3 -m data_gen.gen_test_train_data.gen_from_tweets \
     -twitter_input basic_data/microposts2016/training/ \
     -out_file generated/test_train_data/microposts2016-train.csv
```

Example `output.csv`:
```
675572272418021377	675572272418021377	Etihad Airways	RT @VisitAbuDhabi: You could win 3 days #InAbuDhabi with us & @	, VIP #StarWars Premiere tickets & more! Join now: https://t.co…	CANDIDATES	EMPTYCAND	GT:	-1,512231,Etihad_Airways
675572272418021377	675572272418021377	StarWars	RT @VisitAbuDhabi: You could win 3 days #InAbuDhabi with us & @EtihadAirways, VIP #	Premiere tickets & more! Join now: https://t.co…	CANDIDATES	26678,0.832,Star Wars	52549,0.088,Star Wars (film)	944478,0.044,Star Wars (1983 video game)	29186,0.036,Strategic Defense Initiative	GT:	1,26678,0.832,Star Wars
675572272418021377	675572272418021377	InAbuDhabi	RT @VisitAbuDhabi: You could win 3 days #	with us & @EtihadAirways, VIP #StarWars Premiere tickets & more! Join now: https://t.co…	CANDIDATES	EMPTYCAND	GT:	-1,18950756,Abu_Dhabi
675572272418021377	675572272418021377	VisitAbuDhabi	RT @	: You could win 3 days #InAbuDhabi with us & @EtihadAirways, VIP #StarWars Premiere tickets & more! Join now: https://t.co…	CANDIDATES	EMPTYCAND	GT:	-1,18950756,Abu_Dhabi
```

4) Generate conll files for each of the splits folders (example for training split):
 ```
 python3 -m data_gen.conll_from_tweets -root_data_dir $DATA_PATH \
    -twitter_input basic_data/microposts2016/training/ \
    -out_file generated/microposts2016-train.conll
 ```
 
Example `output.conll`:
```
-DOCSTART- (675572272418021377
RT
VisitAbuDhabi
You
could
win
3
days
InAbuDhabi      B       InAbuDhabi      Abu_Dhabi       http://en.wikipedia.org/wiki/Abu_Dhabi  000     000
with
us
Etihad   B       EtihadAirways   Etihad_Airways  http://en.wikipedia.org/wiki/Etihad_Airways     000     000
Airways  I       EtihadAirways   Etihad_Airways  http://en.wikipedia.org/wiki/Etihad_Airways     000     000
VIP
StarWars        B       StarWars        Star_Wars       http://en.wikipedia.org/wiki/Star_Wars  000     000
Premiere
tickets
more
Join
now
https
t
co
```

TwitterNEED data references: \
[1] Locke, B. and Martin, J. (2009). Named entity recognition: Adapting to microblogging. Senior Thesis, University of Colorado. \
[2] Habib, M. B. and van Keulen, M. (2012). Unsupervised improvement of named entity extraction in short informal context using disambiguation clues. In Proceedings of the Workshop on Semantic Web and Information Extraction (SWAIE 2012), pages 1–10. \
[3] A. E. Cano Basave, G. Rizzo, A. Varga, M. Rowe, M. Stankovic, and A.-S. Dadzie (2014). Making Sense of Microposts (#Microposts2014) Named Entity Extraction & Linking Challenge. In Proceedings of #Microposts2014, pages 54–60, 2014.
