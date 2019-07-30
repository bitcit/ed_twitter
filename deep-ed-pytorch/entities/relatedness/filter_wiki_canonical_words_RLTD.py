# python3 -m entities.relatedness.filter_wiki_canonical_words_RLTD \
# -root_data_dir small-data/


import argparse
import os

from entities.relatedness.relatedness import Relatedness

parser = argparse.ArgumentParser()
parser.add_argument('-root_data_dir', default='data/',
                    help='Root path of the data, $DATA_PATH.')
parser.add_argument('-rltd_test_txt',
                    default='basic_data/relatedness/test.svm')
parser.add_argument('-rltd_val_txt',
                    default='basic_data/relatedness/validate.svm')
parser.add_argument('-rltd_test_dict',
                    default='generated/relatedness_test.dict')
parser.add_argument('-rltd_val_dict',
                    default='generated/relatedness_validate.dict')
parser.add_argument('-rltd_dict',
                    default='generated/all_candidate_ents_ed_rltd_RLTD.dict')
parser.add_argument('-test_train', default='generated/test_train_data/')
parser.add_argument('-wiki_can_words',
                    default='generated/wiki_canonical_words.txt',
                    help='Words from canonical wikipedia pages.')
parser.add_argument('-wiki_can_words_rltd',
                    default='generated/wiki_canonical_words_RLTD.txt')
args = parser.parse_args()

rltd = Relatedness(args)

ouf_path = os.path.join(args.root_data_dir, args.wiki_can_words_rltd)
ouf = open(ouf_path, 'w')

ent_wikiid = -1

num_lines = 0
print('Starting dataset filtering.')
wiki_context = os.path.join(args.root_data_dir, args.wiki_can_words)
with open(wiki_context, 'r') as f:
    for line in f:
        line = line.rstrip()
        num_lines += 1
        if num_lines % 500000 == 0:
            print('Processed {} lines.'.format(num_lines))
        parts = line.split('\t')

        assert len(parts) == 3

        ent_wikiid = int(parts[0])
        ent_name = parts[1]
        assert ent_wikiid

        if ent_wikiid in rltd.wikiid_to_rltdid:
            ouf.write(line + '\n')

ouf.close()
