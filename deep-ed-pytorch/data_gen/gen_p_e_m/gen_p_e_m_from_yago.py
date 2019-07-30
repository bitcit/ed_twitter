# Generate p(e|m) YAGO index from AIDA means
#
# python3 -m data_gen.gen_p_e_m.gen_p_e_m_from_yago \
# -root_data_dir small-data/ \
# -wiki_name_id_map basic_data/wiki_name_id_apple_inc.txt \
# -wiki_disambiguation basic_data/wiki_disambiguation_pages_small.txt \
# -wiki_redirects basic_data/wiki_redirects_apple_inc.txt \
# -aida_means basic_data/p_e_m_data/aida_means_apple_inc.tsv \
# -yago_p_e_m generated/yago_p_e_m_apple_inc.txt

import argparse
import os

from entities.ent_name2id_freq.ent_name_id import EntityNameId
from utils.utils import unicode2ascii

parser = argparse.ArgumentParser()
parser.add_argument('-root_data_dir', default='data/',
                    help='Root path of the data, $DATA_PATH.')
parser.add_argument('-wiki_name_id_map',
                    default='basic_data/wiki_name_id_map.txt',
                    help='Wikipedia name id map.')
parser.add_argument('-wiki_disambiguation',
                    default='basic_data/wiki_disambiguation_pages.txt',
                    help='Wikipedia disambiguation index.')
parser.add_argument('-wiki_redirects',
                    default='basic_data/wiki_redirects.txt',
                    help='Wikipedia redirects index.')
parser.add_argument('-aida_means',
                    default='basic_data/p_e_m_data/aida_means.tsv',
                    help='AIDA means.')
parser.add_argument('-yago_p_e_m',
                    default='generated/yago_p_e_m.txt',
                    help='YAGO entity-mention prior output.')
args = parser.parse_args()

assert os.path.isdir(args.root_data_dir), \
    'Specify a valid root_data_dir path argument.'

entity_name_id = EntityNameId(args)

print('Computing YAGO p_e_m')

aida_means_path = os.path.join(args.root_data_dir, args.aida_means)

num_lines = 0
wiki_e_m_counts = {}

with open(aida_means_path, 'r') as f:
    for line in f:
        num_lines += 1
        if num_lines % 5000000 == 0:
            print('Processed {} lines.'.format(num_lines))
        line = line.rstrip()
        parts = line.split('\t')
        assert len(parts) == 2
        assert parts[0][0] == '"'
        assert parts[0][-1] == '"'
        mention = parts[0][1: -1]
        ent_name = parts[1].rstrip()

        ent_name = ent_name.replace('&amp;', '&')
        ent_name = ent_name.replace('&quot;', '"')
        x = ent_name.find('\\u')
        while x != -1:
            code = ent_name[x:x + 6]
            try:
                replace = unicode2ascii[code]
            except KeyError:
                print('Code {} in line {} error'.format(code, num_lines))
                try:
                    replace = unicode2ascii[code[1:]]
                except KeyError:
                    print('Code {} not in unicode2ascii dict'.format(code))

            if replace == "%":
                replace = "%%"

            ent_name = ent_name.replace(code, replace)
            x = ent_name.find('\\u')

        ent_name = entity_name_id.preprocess_ent_name(ent_name)
        ent_wikiid = entity_name_id.get_ent_wikiid_from_name(ent_name, True)

        if ent_wikiid != entity_name_id.unk_ent_wikiid:
            if mention not in wiki_e_m_counts:
                wiki_e_m_counts[mention] = {}
            wiki_e_m_counts[mention][ent_wikiid] = 1

print('Now writing ..')
out_file = os.path.join(args.root_data_dir, args.yago_p_e_m)

with open(out_file, 'w') as f:
    for mention in wiki_e_m_counts:
        entity_count = ''
        total_count = 0
        for ent_wikiid in wiki_e_m_counts[mention]:
            entity_count += str(ent_wikiid) + ',' + entity_name_id \
                .ent_from_wikiid(ent_wikiid).replace(' ', '_') + '\t'
            total_count += 1
        # Remove trailing whitespace.
        entity_count = entity_count[:-1]
        f.write('{}\t{}\t{}\n'.format(mention, total_count, entity_count))

print('    Done writing.')
