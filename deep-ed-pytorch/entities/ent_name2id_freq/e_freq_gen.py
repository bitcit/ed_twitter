# Creates a file that contains entity counts.
#
# python3 -m entities.ent_name2id_freq.e_freq_gen \
# -root_data_dir small-data/ \
# -wiki_name_id_map basic_data/wiki_name_id_apple_inc.txt \
# -wiki_disambiguation basic_data/wiki_disambiguation_pages_small.txt \
# -wiki_redirects basic_data/wiki_redirects_apple_inc.txt \
# -merged_p_e_m generated/crosswikis_wikipedia_p_e_m_apple_inc.txt \
# -ent_counts generated/ent_wiki_counts_apple_inc.txt

import argparse
import os

from entities.ent_name2id_freq.ent_name_id import EntityNameId

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
parser.add_argument('-merged_p_e_m',
                    default='generated/crosswikis_wikipedia_p_e_m.txt',
                    help='Merged entity-mention prior.')
parser.add_argument('-ent_counts',
                    default='generated/ent_wiki_counts.txt',
                    help='Sorted entity counts output.')
args = parser.parse_args()

assert os.path.isdir(args.root_data_dir), \
    'Specify a valid root_data_dir path argument.'

entity_name_id = EntityNameId(args)

entity_counts = {}

num_lines = 0

merged_p_e_m_path = os.path.join(args.root_data_dir, args.merged_p_e_m)
with open(merged_p_e_m_path, 'r') as f:
    for line in f:
        line = line.rstrip()
        num_lines += 1
        if num_lines % 2000000 == 0:
            print('Processed {} lines.'.format(num_lines))
        parts = line.split('\t')
        num_parts = len(parts)
        for i in range(2, num_parts):
            ent_str = parts[i].split(',')
            wikiid = int(ent_str[0])
            cnt = int(ent_str[1])

            if wikiid not in entity_counts:
                entity_counts[wikiid] = 0

            entity_counts[wikiid] += cnt

#  Writing word frequencies
print('Sorting and writing')

filtered_ec = [(k, v) for k, v in entity_counts.items() if v >= 10]
sorted_ec = sorted(filtered_ec, key=lambda kv: kv[1], reverse=True)

total_count = 0

out_file = os.path.join(args.root_data_dir, args.ent_counts)
with open(out_file, 'w') as f:
    for wikiid, cnt in sorted_ec:
        f.write('{}\t{}\t{}\n'
                .format(wikiid, entity_name_id
                        .ent_from_wikiid(wikiid).replace(' ', '_'), cnt))
        total_count += cnt
print('Total freq = {}'.format(total_count))
