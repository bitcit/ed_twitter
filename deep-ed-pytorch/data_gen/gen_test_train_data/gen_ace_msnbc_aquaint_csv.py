# Generate test data from the ACE/MSNBC/AQUAINT datasets by keeping the
# Generate test data from the ACE/MSNBC/AQUAINT datasets by keeping the
# context and entity candidates for each annotated mention
#
# Format:
# doc_name \t doc_name \t mention \t left_ctxt \t right_ctxt \t CANDIDATES \t
# [ent_wikiid,p_e_m,ent_name]+ \t GT: \t pos,ent_wikiid,p_e_m,ent_name

# Stats:
# wc -l $DATA_PATH/generated/test_train_data/wned-ace2004.csv
# 257
# grep -P 'GT:\t-1' $DATA_PATH/generated/test_train_data/wned-ace2004.csv | wc -l
# 20
# grep -P 'GT:\t1,' $DATA_PATH/generated/test_train_data/wned-ace2004.csv | wc -l
# 217
#
# wc -l $DATA_PATH/generated/test_train_data/wned-aquaint.csv
# 727
# grep -P 'GT:\t-1' $DATA_PATH/generated/test_train_data/wned-aquaint.csv | wc -l
# 33
# grep -P 'GT:\t1,' $DATA_PATH/generated/test_train_data/wned-aquaint.csv | wc -l
# 604
#
# wc -l $DATA_PATH/generated/test_train_data/wned-msnbc.csv
# 656
# grep -P 'GT:\t-1' $DATA_PATH/generated/test_train_data/wned-msnbc.csv | wc -l
# 22
# grep -P 'GT:\t1,' $DATA_PATH/generated/test_train_data/wned-msnbc.csv | wc -l
# 496

import os
from xml.etree import ElementTree

from utils.utils import split_in_words


def gen_test_ace(wned_path, test_train_path, dataset, yago_crw_index,
                 entity_name_id):
    print('Generating test data from ' + dataset + ' set ')
    annotations_xml = os.path.join(wned_path,
                                   '{}/{}.xml'.format(dataset, dataset))
    ouf_path = os.path.join(test_train_path, 'wned-{}.csv'.format(dataset))
    ouf = open(ouf_path, 'w')

    num_nonexistent_ent_id = 0
    num_correct_ents = 0

    tree = ElementTree.parse(annotations_xml)
    root = tree.getroot()

    for doc in root:
        doc_name = doc.attrib['docName'].replace('&amp;', '&')
        # print('doc_name: {}'.format(doc_name))
        doc_path = os.path.join(wned_path, '{}/RawText/{}'
                                .format(dataset, doc_name))
        with open(doc_path, 'r') as cf:
            doc_text = ' '.join(cf.readlines())
        doc_text = doc_text.replace('&amp;', '&')

        for annotation in doc:
            mention = annotation.find('mention').text.replace('&amp;', '&')
            ent_title = annotation.find('wikiName').text
            offset = int(annotation.find('offset').text)  # + 1
            length = int(annotation.find('length').text)  # len(mention)

            # TODO: wtf?
            offset = max(0, offset - 10)

            offset = doc_text.find(mention, offset)
            assert offset >= 0, 'Mention {} not found in doc {} at offset {}' \
                .format(mention, doc_name, offset)
            # while doc_text[offset: offset + length] != mention:
            #     offset += 1

            mention = yago_crw_index.preprocess_mention(mention)

            if not ent_title or ent_title == 'NIL':
                continue

            cur_ent_wikiid = entity_name_id.get_ent_wikiid_from_name(ent_title)
            if cur_ent_wikiid == entity_name_id.unk_ent_wikiid:
                num_nonexistent_ent_id += 1
                print(ent_title)  # green
            else:
                num_correct_ents += 1

            ctxt_str = '\t'.join([doc_name, doc_name, mention]) + '\t'

            left_words = split_in_words(doc_text[:offset])

            start = max(0, len(left_words) - 100)
            left_ctxt = left_words[start:]
            if not left_ctxt:
                left_ctxt.append('EMPTYCTXT')

            ctxt_str += ' '.join(left_ctxt) + '\t'

            right_words = split_in_words(doc_text[offset + length:])

            end = min(len(right_words), 100)
            right_ctxt = right_words[:end]
            if not right_ctxt:
                right_ctxt.append('EMPTYCTXT')

            ctxt_str += ' '.join(right_ctxt) + '\tCANDIDATES\t'

            # Entity candidates from p(e|m) dictionary
            if mention in yago_crw_index.ent_p_e_m_index and \
                    len(yago_crw_index.ent_p_e_m_index[mention]) > 0:
                sorted_cand = {}
                for ent_wikiid in yago_crw_index.ent_p_e_m_index[mention]:
                    sorted_cand[ent_wikiid] = \
                        yago_crw_index.ent_p_e_m_index[mention][ent_wikiid]

                sorted_cand = sorted(sorted_cand.items(), key=lambda kv: kv[1],
                                     reverse=True)
                gt_pos = -1
                pos = 0
                candidates = []
                for ent_wikiid, p in sorted_cand:
                    if pos > 100:
                        break

                    candidates.append('{},{:.3f},{}'
                                      .format(ent_wikiid, p, entity_name_id
                                              .ent_from_wikiid(ent_wikiid)))

                    pos += 1
                    if ent_wikiid == cur_ent_wikiid:
                        gt_pos = pos

                ctxt_str += '\t'.join(candidates) + '\tGT:\t'

                if gt_pos > 0:
                    ouf.write('{}{},{}\n'.format(ctxt_str, gt_pos,
                                                 candidates[gt_pos - 1]))
                elif cur_ent_wikiid != entity_name_id.unk_ent_wikiid:
                    ouf.write('{}-1{},{}\n'
                              .format(ctxt_str, cur_ent_wikiid, ent_title))
                else:
                    ouf.write(ctxt_str + '-1\n')
            elif cur_ent_wikiid != entity_name_id.unk_ent_wikiid:
                ouf.write('{}EMPTYCAND\tGT:\t-1,{},{}\n'
                          .format(ctxt_str, cur_ent_wikiid, ent_title))
            else:
                ouf.write('{}EMPTYCAND\tGT:\t-1\n'.format(ctxt_str))

    ouf.close()
    print('Done {}.'.format(dataset))
    print('num_nonexistent_ent_id = {}; num_correct_ents = {}'
          .format(num_nonexistent_ent_id, num_correct_ents))
