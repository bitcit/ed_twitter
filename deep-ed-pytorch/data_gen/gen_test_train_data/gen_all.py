# Generates all training and test data for entity disambiguation.
#
# python3 -m data_gen.gen_test_train_data.gen_all \
# -root_data_dir small-data/ \
# -wiki_name_id_map basic_data/wiki_name_id_apple_inc.txt \
# -wiki_disambiguation basic_data/wiki_disambiguation_pages_small.txt \
# -wiki_redirects basic_data/wiki_redirects_apple_inc.txt \
# -merged_p_e_m generated/crosswikis_wikipedia_p_e_m_apple_inc.txt \
# -yago_p_e_m generated/yago_p_e_m_apple_inc.txt

import argparse
import os

from data_gen.indexes.yago_crosswikis_wiki import YagoCrosswikisIndex
from entities.ent_name2id_freq.ent_name_id import EntityNameId
from data_gen.gen_test_train_data.gen_aida_test import gen_aida_test
from data_gen.gen_test_train_data.gen_aida_train import gen_aida_train
from data_gen.gen_test_train_data.gen_ace_msnbc_aquaint_csv import gen_test_ace

parser = argparse.ArgumentParser()
parser.add_argument('-root_data_dir', default='data/',
                    help='Root path of the data, $DATA_PATH.')
parser.add_argument('-wiki_name_id_map',
                    default='basic_data/wiki_name_id_map.txt',
                    help='Wikipedia name id map.')
parser.add_argument('-wiki_disambiguation',
                    default='basic_data/wiki_disambiguation_pages.txt',
                    help='Wikipedia disambiguation index.')
parser.add_argument('-wiki_redirects', default='basic_data/wiki_redirects.txt',
                    help='Wikipedia redirects index.')
parser.add_argument('-merged_p_e_m',
                    default='generated/crosswikis_wikipedia_p_e_m.txt',
                    help='Merged entity-mention prior.')
parser.add_argument('-yago_p_e_m', default='generated/yago_p_e_m.txt',
                    help='YAGO entity-mention prior output.')
parser.add_argument('-aida_test',
                    default='basic_data/test_datasets/AIDA/testa_testb_aggregate_original')
parser.add_argument('-aida_a',
                    default='generated/test_train_data/aida_testA.csv')
parser.add_argument('-aida_b',
                    default='generated/test_train_data/aida_testB.csv')
parser.add_argument('-aida_train',
                    default='basic_data/test_datasets/AIDA/aida_train.txt')
parser.add_argument('-aida_train_out',
                    default='generated/test_train_data/aida_train.csv')
parser.add_argument('-wned', default='basic_data/test_datasets/wned-datasets/')
parser.add_argument('-test_train', default='generated/test_train_data/')
args = parser.parse_args()

entity_name_id = EntityNameId(args)

# TODO: all three scripts share significant amount of code.
crosswikis_path = os.path.join(args.root_data_dir, args.merged_p_e_m)
yago_p_e_m_path = os.path.join(args.root_data_dir, args.yago_p_e_m)
yago_crosswikis_index = YagoCrosswikisIndex(crosswikis_path, yago_p_e_m_path)

aida_test_path = os.path.join(args.root_data_dir, args.aida_test)
aida_a_path = os.path.join(args.root_data_dir, args.aida_a)
aida_b_path = os.path.join(args.root_data_dir, args.aida_b)
gen_aida_test(aida_test_path, aida_a_path, aida_b_path, yago_crosswikis_index,
              entity_name_id)

aida_train_path = os.path.join(args.root_data_dir, args.aida_train)
aida_train_out_path = os.path.join(args.root_data_dir, args.aida_train_out)
gen_aida_train(aida_train_path, aida_train_out_path, yago_crosswikis_index,
               entity_name_id)

wned_path = os.path.join(args.root_data_dir, args.wned)
test_train_path = os.path.join(args.root_data_dir, args.test_train)
gen_test_ace(wned_path, test_train_path, 'wikipedia', yago_crosswikis_index,
             entity_name_id)
gen_test_ace(wned_path, test_train_path, 'clueweb', yago_crosswikis_index,
             entity_name_id)
gen_test_ace(wned_path, test_train_path, 'ace2004', yago_crosswikis_index,
             entity_name_id)
gen_test_ace(wned_path, test_train_path, 'msnbc', yago_crosswikis_index,
             entity_name_id)
gen_test_ace(wned_path, test_train_path, 'aquaint', yago_crosswikis_index,
             entity_name_id)
