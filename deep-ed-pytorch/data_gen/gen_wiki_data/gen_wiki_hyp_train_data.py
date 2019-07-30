# Generate training data from Wikipedia hyperlinks by keeping the context and
# entity candidates for each hyperlink

# python3 -m data_gen.gen_wiki_data.gen_wiki_hyp_train_data \
# -root_data_dir small-data/ \
# -wiki_text basic_data/textWithAnchorsFromAppleInc.txt \
# -wiki_name_id_map basic_data/wiki_name_id_apple_inc.txt \
# -wiki_disambiguation basic_data/wiki_disambiguation_pages_small.txt \
# -wiki_redirects basic_data/wiki_redirects_apple_inc.txt \
# -merged_p_e_m generated/crosswikis_wikipedia_p_e_m_apple_inc.txt \
# -yago_p_e_m generated/yago_p_e_m_apple_inc.txt

import argparse
import os

from data_gen.parse_wiki_dump.parse_wiki_dump_tools import \
    extract_text_and_hyp, extract_page_entity_title
from entities.ent_name2id_freq.ent_name_id import EntityNameId
from data_gen.indexes.yago_crosswikis_wiki import YagoCrosswikisIndex
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
parser.add_argument('-merged_p_e_m',
                    default='generated/crosswikis_wikipedia_p_e_m.txt',
                    help='Merged entity-mention prior.')
parser.add_argument('-yago_p_e_m', default='generated/yago_p_e_m.txt',
                    help='YAGO entity-mention prior output.')
parser.add_argument('-wiki_ctxt',
                    default='generated/wiki_hyperlink_contexts.csv')
args = parser.parse_args()

# Format:
# ent_wikiid \t ent_name \t mention \t left_ctxt \t right_ctxt \t CANDIDATES \t
# [ent_wikiid,p_e_m,ent_name]+ \t GT: \t pos,ent_wikiid,p_e_m,ent_name

num_lines = 0
num_valid_hyperlinks = 0
missing_wikiids = 0

cur_words = []
cur_mentions = []
cur_ent_wikiid = -1

e_name_id = EntityNameId(args)

crosswikis_path = os.path.join(args.root_data_dir, args.merged_p_e_m)
yago_p_e_m_path = os.path.join(args.root_data_dir, args.yago_p_e_m)
yago_cws_index = YagoCrosswikisIndex(crosswikis_path, yago_p_e_m_path)

ouf_path = os.path.join(args.root_data_dir, args.wiki_ctxt)
ouf = open(ouf_path, 'w')

print('Generating training data from Wiki dump')
wiki_path = os.path.join(args.root_data_dir, args.wiki_text)
with open(wiki_path, 'r') as f:
    for line in f:
        line = line.rstrip()
        num_lines = num_lines + 1
        if num_lines % 1000000 == 0:
            print('Processed {} lines. Num valid hyperlinks = {}'
                  .format(num_lines, num_valid_hyperlinks))

        # Parse wiki text and hyperlinks.
        if '<doc id="' not in line and '</doc>' not in line:
            list_hyp, text, _, _, _, _ = extract_text_and_hyp(line, True,
                                                              e_name_id)
            words = split_in_words(text)

            line_mentions = {}
            num_added_hyp = 0
            for w in words:
                if w[:len('MMSTART')] == 'MMSTART':
                    mention_idx = int(w[len('MMSTART'):])
                    assert mention_idx, w
                    line_mentions[mention_idx] = {'start_off': len(cur_words),
                                                  'end_off': -1}
                elif w[:len('MMEND')] == 'MMEND':
                    num_added_hyp += 1
                    mention_idx = int(w[len('MMEND'):])
                    assert mention_idx, w
                    assert mention_idx in line_mentions
                    line_mentions[mention_idx]['end_off'] = len(cur_words)
                else:
                    cur_words.append(w)

            assert len(list_hyp) == num_added_hyp, '{} :: {} :: {} {}' \
                .format(line, text, num_added_hyp, len(list_hyp))

            for el in list_hyp:
                cur_mentions \
                    .append({'mention': el['mention'],
                             'ent_wikiid': el['ent_wikiid'],
                             'start_off': line_mentions[el['cnt']]['start_off'],
                             'end_off': line_mentions[el['cnt']]['end_off']})

        elif '<doc id="' in line:
            if cur_ent_wikiid != e_name_id.unk_ent_wikiid and \
                    e_name_id.is_valid_ent(cur_ent_wikiid):

                header = '{}\t{}\t'.format(cur_ent_wikiid, e_name_id
                                           .ent_from_wikiid(cur_ent_wikiid))
                for hyp in cur_mentions:
                    mention = hyp['mention']
                    assert len(mention) > 0, line

                    if mention not in yago_cws_index.ent_p_e_m_index or \
                            len(yago_cws_index.ent_p_e_m_index[mention]) < 1:
                        continue

                    ctxt_str = header + mention + '\t'

                    start = max(0, hyp['start_off'] - 101)
                    end = hyp['start_off'] - 1
                    left_ctxt = cur_words[start: end]
                    if not left_ctxt:
                        left_ctxt.append('EMPTYCTXT')

                    ctxt_str += ' '.join(left_ctxt) + '\t'

                    start = hyp['end_off']
                    end = min(len(cur_words), hyp['end_off'] + 100)
                    right_ctxt = cur_words[start: end]
                    if not right_ctxt:
                        right_ctxt.append('EMPTYCTXT')

                    ctxt_str += ' '.join(right_ctxt) + '\tCANDIDATES\t'

                    # Entity candidates from p(e|m) dictionary
                    sorted_cand = {}
                    for ent_wikiid in yago_cws_index.ent_p_e_m_index[mention]:
                        sorted_cand[ent_wikiid] = \
                            yago_cws_index.ent_p_e_m_index[mention][ent_wikiid]

                    sorted_cand = sorted(sorted_cand.items(),
                                         key=lambda kv: kv[1], reverse=True)

                    gt_pos = -1
                    pos = 0
                    candidates = []
                    for ent_wikiid, p in sorted_cand:
                        if pos > 32:
                            break

                        candidates.append('{},{:.3f},{}'
                                          .format(ent_wikiid, p, e_name_id
                                                  .ent_from_wikiid(ent_wikiid)))

                        pos += 1
                        if ent_wikiid == hyp['ent_wikiid']:
                            gt_pos = pos

                    ctxt_str += '\t'.join(candidates) + '\tGT:\t'

                    if gt_pos > 0:
                        num_valid_hyperlinks += 1
                        ouf.write('{}{},{}\n'.format(ctxt_str, gt_pos,
                                                     candidates[gt_pos - 1]))

            cur_ent_wikiid, missing = extract_page_entity_title(line, e_name_id)
            missing_wikiids += missing

            cur_words = []
            cur_mentions = []

ouf.close()
print('    Done generating training data from Wiki dump. Num valid hyp = {}. '
      'Num missing wikiids {}.'.format(num_valid_hyperlinks, missing_wikiids))
