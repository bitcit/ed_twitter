#!/bin/bash

export DATA_PATH=data/
mkdir -p $DATA_PATH
mkdir -p $DATA_PATH/generated/

# UNCOMMENT THE FOLLOWING LINES TO DOWNLOAD DATA
# Download basic data
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}
gdrive_download 0Bx8d3azIm_ZcbHMtVmRVc1o5TWM $DATA_PATH/basic_data.zip
unzip -q $DATA_PATH/basic_data.zip -d $DATA_PATH

# Download word2vec
gdrive_download 0B7XkCwpI5KDYNlNUTTlSS21pQmM $DATA_PATH/basic_data/wordEmbeddings/Word2Vec/GoogleNews-vectors-negative300.bin.gz
gunzip $DATA_PATH/basic_data/wordEmbeddings/Word2Vec/GoogleNews-vectors-negative300.bin.gz

# Pre-process data
python3 -m data_gen.gen_p_e_m.gen_p_e_m_from_wiki -root_data_dir $DATA_PATH |& tee 1_py.out
python3 -m data_gen.gen_p_e_m.merge_crosswikis_wiki -root_data_dir $DATA_PATH |& tee 2_py.out
python3 -m data_gen.gen_p_e_m.gen_p_e_m_from_yago -root_data_dir $DATA_PATH |& tee 3_py.out
python3 -m entities.ent_name2id_freq.e_freq_gen -root_data_dir $DATA_PATH |& tee 4_py.out
mkdir -p $DATA_PATH/generated/test_train_data/
python3 -m data_gen.gen_test_train_data.gen_all -root_data_dir $DATA_PATH |& tee 5_py.out
python3 -m data_gen.gen_wiki_data.gen_ent_wiki_w_repr -root_data_dir $DATA_PATH |& tee 6_py.out
python3 -m data_gen.gen_wiki_data.gen_wiki_hyp_train_data -root_data_dir $DATA_PATH |& tee 7_py.out
python3 -m words.w_freq.w_freq_gen -root_data_dir $DATA_PATH |& tee 8_py.out
python3 -m entities.relatedness.filter_wiki_canonical_words_RLTD -root_data_dir $DATA_PATH |& tee 9_py.out
python3 -m entities.relatedness.filter_wiki_hyperlink_contexts_RLTD -root_data_dir $DATA_PATH |& tee 10_py.out

# Train entity embeddings
mkdir -p $DATA_PATH/generated/ent_vecs
python3 -u -m entities.learn_e2v.learn_a -root_data_dir $DATA_PATH |& tee log_train_entity_vecs
