#!/bin/bash

# Create directory and subdirectories for "medium size dataset"
mkdir data_m && mkdir data_m/basic_data && mkdir data_m/basic_data/p_e_m_data
mkdir data_m/basic_data/relatedness && mkdir data_m/basic_data/test_datasets
mkdir data_m/generated
# Extract first 144 Wikipedia articles.
sed -n 1,9459p data/basic_data/textWithAnchorsFromAllWikipedia2014Feb.txt > data_m/basic_data/textWithAnchorsFromAllWikipedia2014Feb.txt
# Keep all entities (article titles) in a document.
grep -Po 'title="\K[^"]*' data_m/basic_data/textWithAnchorsFromAllWikipedia2014Feb.txt > data_m/entities.txt

declare -a StringArray=("wiki_name_id_map.txt" "wiki_disambiguation_pages.txt"
"wiki_redirects.txt" "p_e_m_data/aida_means.tsv"
"p_e_m_data/crosswikis_p_e_m.txt" )

while IFS='' read -r entity || [[ -n "$entity" ]]; do
    for f in "${StringArray[@]}"; do
        grep "$entity" data/basic_data/"$f" >> data_m/basic_data/"$f"
    done
done < data_m/entities.txt

head data/basic_data/relatedness/test.svm -n 1000 > data_m/basic_data/relatedness/test.svm
head data/basic_data/relatedness/validate.svm -n 1000 > data_m/basic_data/relatedness/validate.svm
cp -r data/basic_data/test_datasets data_m/basic_data/
