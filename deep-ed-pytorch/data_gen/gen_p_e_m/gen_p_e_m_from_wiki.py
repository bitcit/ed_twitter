# Generate entity-mention prior from Wikipedia
#
# python3 -m data_gen.gen_p_e_m.gen_p_e_m_from_wiki \
# -root_data_dir small-data/ \
# -wiki_text basic_data/textWithAnchorsFromAppleInc.txt \
# -wiki_name_id_map basic_data/wiki_name_id_apple_inc.txt \
# -wiki_disambiguation basic_data/wiki_disambiguation_pages_small.txt \
# -wiki_redirects basic_data/wiki_redirects_apple_inc.txt \
# -wiki_p_e_m generated/wikipedia_p_e_m_apple_inc.txt

import argparse
import os

from data_gen.parse_wiki_dump.parse_wiki_dump_tools import \
    extract_text_and_hyp
from entities.ent_name2id_freq.ent_name_id import EntityNameId

parser = argparse.ArgumentParser()
parser.add_argument('-root_data_dir', default='data/',
                    help='Root path of the data, $DATA_PATH.')
parser.add_argument('-wiki_text', default='basic_data/textWithAnchorsFromAll'
                                          'Wikipedia2014Feb.txt',
                    help='Output from wiki extractor.')
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
                    help='Entity-mention prior output based on wikipedia.')
args = parser.parse_args()

assert os.path.isdir(args.root_data_dir), \
    'Specify a valid root_data_dir path argument.'

# Find anchors, e.g. <a href="wikt:anarchism">anarchism</a>
num_lines = 0
parsing_errors = 0
list_ent_errors = 0
diez_ent_errors = 0
disambiguation_ent_errors = 0
num_valid_hyperlinks = 0

wiki_e_m_counts = {}

entity_name_id = EntityNameId(args)

wiki_path = os.path.join(args.root_data_dir, args.wiki_text)

print('    Computing Wikipedia p_e_m.')
with open(wiki_path, 'r') as f:
    for line in f:
        line = line.rstrip()
        num_lines = num_lines + 1
        if num_lines % 500000 == 0:
            print('Processed {} lines. Parsing errs = {} List ent errs = {} '
                  'diez errs = {} disambig errs = {}.'
                  'Num valid hyperlinks = {}.'
                  .format(num_lines, parsing_errors, list_ent_errors,
                          diez_ent_errors, disambiguation_ent_errors,
                          num_valid_hyperlinks))

        # Parse wiki text and hyperlinks.
        if '<doc id="' in line:
            continue

        list_hyp, text, le_errs, p_errs, dis_errs, diez_errs = \
            extract_text_and_hyp(line, False, entity_name_id)
        parsing_errors += p_errs
        list_ent_errors += le_errs
        disambiguation_ent_errors += dis_errs
        diez_ent_errors += diez_errs

        for el in list_hyp:
            mention = el['mention']
            ent_wikiid = el['ent_wikiid']

            # A valid (entity,mention) pair
            num_valid_hyperlinks += 1

            if mention not in wiki_e_m_counts:
                wiki_e_m_counts[mention] = {}

            if ent_wikiid not in wiki_e_m_counts[mention]:
                wiki_e_m_counts[mention][ent_wikiid] = 0

            wiki_e_m_counts[mention][ent_wikiid] += 1

# Num valid ents = 4126137. Num errs = 332944
print('    Done computing Wikipedia p(e|m). Num valid hyperlinks = {}, '
      'num lines {}'.format(num_valid_hyperlinks, num_lines))

print('Now sorting and writing ..')
out_file = os.path.join(args.root_data_dir, args.wiki_p_e_m)

with open(out_file, 'w') as f:
    for mention in wiki_e_m_counts:
        ent_wikiids = sorted(wiki_e_m_counts[mention].items(),
                             key=lambda kv: kv[1], reverse=True)

        entity_counts = ''
        total_count = 0
        for ent_wikiid, count in ent_wikiids:
            entity_counts += '{},{},{}\t' \
                .format(ent_wikiid, count, entity_name_id
                        .ent_from_wikiid(ent_wikiid).replace(' ', '_'))
            total_count += count
        # Remove trailing whitespace.
        entity_counts = entity_counts[:-1]
        f.write('{}\t{}\t{}\n'.format(mention, total_count, entity_counts))

print('    Done sorting and writing.')
