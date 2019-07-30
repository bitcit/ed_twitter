# Generate training data from Wikipedia canonical pages.
#
# python3 -m data_gen.gen_wiki_data.gen_ent_wiki_w_repr \
# -root_data_dir small-data/ \
# -wiki_text basic_data/textWithAnchorsFromAppleInc.txt \
# -wiki_name_id_map basic_data/wiki_name_id_apple_inc.txt \
# -wiki_disambiguation basic_data/wiki_disambiguation_pages_small.txt \
# -wiki_redirects basic_data/wiki_redirects_apple_inc.txt \
# -ent_counts generated/ent_wiki_counts_apple_inc.txt

import argparse
import os

from data_gen.parse_wiki_dump.parse_wiki_dump_tools import \
    extract_text_and_hyp, extract_page_entity_title
from entities.ent_name2id_freq.ent_name_id import EntityNameId
from entities.ent_name2id_freq.e_freq_index import EntityCountMap
from utils.utils import split_in_words

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
parser.add_argument('-wiki_can_words',
                    default='generated/wiki_canonical_words.txt',
                    help='Words from canonical wikipedia pages.')
parser.add_argument('-empty_p_ent',
                    default='generated/empty_page_ents.txt',
                    help='Entities without canonical wikipedia pages.')
parser.add_argument('-ent_counts',
                    default='generated/ent_wiki_counts.txt',
                    help='Sorted entity counts.')
args = parser.parse_args()

assert os.path.isdir(args.root_data_dir), \
    'Specify a valid root_data_dir path argument.'

# Find anchors, e.g. <a href="wikt:anarchism">anarchism</a>
num_lines = 0

num_valid_ents = 0
num_error_ents = 0  # Probably list or disambiguation pages.
num_double_docids = 0

e_name_id = EntityNameId(args)

cur_ent_wikiid = -1
cur_words = ''
empty_valid_ents = e_name_id.get_map_all_valid_ents()

wiki_canonical_words = os.path.join(args.root_data_dir, args.wiki_can_words)
can_words_f = open(wiki_canonical_words, "w")

wiki_path = os.path.join(args.root_data_dir, args.wiki_text)
missing_wikiids = 0

print('    Extracting text from Wiki dump containing on each line an Wiki '
      'entity with the list of all words from its canonical Wiki page.')
with open(wiki_path, 'r') as f:
    for line in f:
        line = line.rstrip()
        num_lines = num_lines + 1
        if num_lines % 500000 == 0:
            print('Processed {} lines. Num valid ents = {} Num errs ents = {}.'
                  .format(num_lines, num_valid_ents, num_error_ents))

        # Parse wiki text and hyperlinks.
        if '<doc id="' not in line and '</doc>' not in line:
            _, text, _, _, _, _ = extract_text_and_hyp(line, False, e_name_id)
            words = split_in_words(text)
            cur_words += ' '.join(words) + ' '

        elif '<doc id="' in line:
            if cur_ent_wikiid >= 0 and cur_words.strip():
                if cur_ent_wikiid != e_name_id.unk_ent_wikiid and \
                        e_name_id.is_valid_ent(cur_ent_wikiid):
                    can_words_f.write(
                        '{}\t{}\t{}\n'.format(cur_ent_wikiid, e_name_id
                                              .ent_from_wikiid(cur_ent_wikiid),
                                              cur_words))
                    if cur_ent_wikiid in empty_valid_ents:
                        empty_valid_ents.pop(cur_ent_wikiid)
                    else:
                        # Double occurence of the same docid in wiki dump
                        num_double_docids += 1
                    num_valid_ents += 1
                else:
                    num_error_ents += 1

            cur_ent_wikiid, missing = extract_page_entity_title(line, e_name_id)
            missing_wikiids += missing
            cur_words = ''

can_words_f.close()
print('All missing wikiids from Wiki dump: {}'.format(missing_wikiids))

# Num valid ents = 4126137. Num errs = 332944
print('    Done extracting text from Wiki dump. Num valid ents = {} '
      'Num errs = {} Num double docids {}'
      .format(num_valid_ents, num_error_ents, num_double_docids))

print('Create file with all entities with empty Wikipedia pages.')
ent_cnt_map = EntityCountMap(os.path.join(args.root_data_dir, args.ent_counts))

empty_ents = {}
for ent_wikiid in empty_valid_ents:
    empty_ents[ent_wikiid] = ent_cnt_map.get_ent_freq(ent_wikiid)

empty_ents = sorted(empty_ents.items(), key=lambda kv: kv[1], reverse=True)

out_file2 = os.path.join(args.root_data_dir, args.empty_p_ent)
with open(out_file2, 'w') as f:
    for k, v in empty_ents:
        f.write('{}\t{}\t{}\n'.format(k, e_name_id.ent_from_wikiid(k), v))
