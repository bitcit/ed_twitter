# Merge Wikipedia and Crosswikis p(e|m) indexes
#
# python3 -m data_gen.gen_p_e_m.merge_crosswikis_wiki \
# -root_data_dir small-data/ \
# -wiki_name_id_map basic_data/wiki_name_id_apple_inc.txt \
# -wiki_disambiguation basic_data/wiki_disambiguation_pages_small.txt \
# -wiki_redirects basic_data/wiki_redirects_apple_inc.txt \
# -wiki_p_e_m generated/wikipedia_p_e_m_apple_inc.txt \
# -crosswikis_p_e_m basic_data/p_e_m_data/crosswikis_p_e_m_apple_inc.txt \
# -merged_p_e_m generated/crosswikis_wikipedia_p_e_m_apple_inc.txt

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
parser.add_argument('-wiki_p_e_m',
                    default='generated/wikipedia_p_e_m.txt',
                    help='Entity-mention prior based on wikipedia.')
parser.add_argument('-crosswikis_p_e_m',
                    default='basic_data/p_e_m_data/crosswikis_p_e_m.txt',
                    help='Entity-mention prior based on crosswikis.')
parser.add_argument('-merged_p_e_m',
                    default='generated/crosswikis_wikipedia_p_e_m.txt',
                    help='Merged entity-mention prior output.')
args = parser.parse_args()

assert os.path.isdir(args.root_data_dir), \
    'Specify a valid root_data_dir path argument.'

e_name_id = EntityNameId(args)


def load_p_e_m(mutable_e_m_counts, prior_path):
    with open(prior_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            parts = line.split('\t')
            mention = parts[0]

            if 'Wikipedia' not in mention and 'wikipedia' not in mention:
                if mention not in mutable_e_m_counts:
                    mutable_e_m_counts[mention] = {}

                num_parts = len(parts)
                for i in range(2, num_parts):
                    ent_str = parts[i].split(',')
                    wikiid = int(ent_str[0])
                    cnt = int(ent_str[1])

                    if wikiid not in mutable_e_m_counts[mention]:
                        mutable_e_m_counts[mention][wikiid] = 0
                    mutable_e_m_counts[mention][wikiid] = \
                        mutable_e_m_counts[mention][wikiid] + cnt
    return mutable_e_m_counts


print('Merging Wikipedia and Crosswikis p_e_m')

merged_e_m_counts = {}

print('Process Wikipedia')
wiki_prior_path = os.path.join(args.root_data_dir, args.wiki_p_e_m)
merged_e_m_counts = load_p_e_m(merged_e_m_counts, wiki_prior_path)
print('Found {} mentions'.format(len(merged_e_m_counts)))

print('Process Crosswikis')
crosswikis_prior_path = os.path.join(args.root_data_dir, args.crosswikis_p_e_m)
merged_e_m_counts = load_p_e_m(merged_e_m_counts, crosswikis_prior_path)
print('Found {} mentions'.format(len(merged_e_m_counts)))

print('Now sorting and writing ..')
out_file = os.path.join(args.root_data_dir, args.merged_p_e_m)

with open(out_file, 'w') as f:
    for ent_mention in merged_e_m_counts:
        if len(ent_mention) < 1:
            continue
        ent_wikiids = sorted(merged_e_m_counts[ent_mention].items(),
                             key=lambda kv: kv[1], reverse=True)

        entity_count = ''
        total_count = 0
        num_ents = 0
        for ent_wikiid, count in ent_wikiids:
            if ent_wikiid in e_name_id.ent_wikiid2name:
                entity_count += '{},{},{}\t' \
                    .format(ent_wikiid, count, e_name_id
                            .ent_from_wikiid(ent_wikiid).replace(' ', '_'))
                total_count += count
                num_ents += 1

                # At most 100 candidates
                if num_ents >= 100:
                    break

        # Remove trailing whitespace.
        entity_count = entity_count[:-1]
        f.write('{}\t{}\t{}\n'.format(ent_mention, total_count, entity_count))

print('    Done sorting and writing.')
