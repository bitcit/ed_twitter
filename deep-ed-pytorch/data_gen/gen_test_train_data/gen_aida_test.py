# Generate test data from the AIDA dataset by keeping the context and
# entity candidates for each annotated mention.

# Output format:
# doc_name \t doc_name \t mention \t left_ctxt \t right_ctxt \t CANDIDATES \t
# [ent_wikiid,p_e_m,ent_name]+ \t GT: \t pos,ent_wikiid,p_e_m,ent_name

# Verify stats:
# wc -l $DATA_PATH/generated/test_train_data/aida_testA.csv
# 4791
# grep -P 'GT:\t-1' $DATA_PATH/generated/test_train_data/aida_testA.csv | wc -l
# 43
# grep -P 'GT:\t1' $DATA_PATH/generated/test_train_data/aida_testA.csv | wc -l
# 3401

# wc -l $DATA_PATH/generated/test_train_data/aida_testB.csv
# 4485
# grep -P 'GT:\t-1' $DATA_PATH/generated/test_train_data/aida_testB.csv | wc -l
# 19
# grep -P 'GT:\t1' $DATA_PATH/generated/test_train_data/aida_testB.csv | wc -l
# 3084

from utils.utils import modify_uppercase_phrase, split_in_words


def gen_aida_test(aida_test, aida_a, aida_b, yago_crosswikis_index,
                  e_name_id):
    print('Generating test data from AIDA set ')
    ouf = open(aida_a, 'w')

    num_nme = 0
    num_nonexistent_ent_title = 0
    num_nonexistent_ent_id = 0
    num_nonexistent_both = 0
    num_correct_ents = 0
    num_total_ents = 0
    cur_words = []
    cur_mentions = []

    cur_doc_name = ''

    with open(aida_test, 'r') as f:
        for line in f:
            line = line.rstrip()
            if '-DOCSTART-' in line:
                write_results(cur_doc_name, cur_mentions, cur_words,
                              yago_crosswikis_index, e_name_id, ouf)

                if 'testa' in cur_doc_name and 'testb' in line:
                    ouf.close()
                    ouf = open(aida_b, 'w')
                    print('Done validation testA : ')
                    print('num_nme = {}; num_nonexistent_ent_title = {}'
                          .format(num_nme, num_nonexistent_ent_title))
                    print('num_nonexistent_ent_id = {}; '
                          'num_nonexistent_both = {}'
                          .format(num_nonexistent_ent_id, num_nonexistent_both))
                    print('num_correct_ents = {}; num_total_ents = {}'
                          .format(num_correct_ents, num_total_ents))

                for w in split_in_words(line):
                    if 'testa' in w or 'testb' in w:
                        cur_doc_name = w
                        break
                cur_words = []
                cur_mentions = []
            else:
                parts = line.split('\t')
                assert len(parts) in [0, 1, 4, 6, 7], line
                if len(parts) <= 0:
                    continue
                if len(parts) == 4 and parts[1] == 'B':
                    num_nme += 1

                if len(parts) in [7, 6] and parts[1] == 'B':
                    # Find current mention. A few hacks here.
                    cur_mention = yago_crosswikis_index \
                        .preprocess_mention(parts[2])

                    y = parts[4].find('/wiki/') + len('/wiki/')
                    cur_ent_title = parts[4][y:]
                    cur_ent_wikiid = int(parts[5])
                    index_ent_title = e_name_id.ent_from_wikiid(cur_ent_wikiid)
                    index_ent_wikiid = e_name_id.get_ent_wikiid_from_name(
                        cur_ent_title)

                    final_ent_wikiid = index_ent_wikiid
                    if final_ent_wikiid == e_name_id.unk_ent_wikiid:
                        final_ent_wikiid = cur_ent_wikiid

                    if index_ent_title == cur_ent_title and \
                            cur_ent_wikiid == index_ent_wikiid:
                        num_correct_ents += 1
                    elif index_ent_title != cur_ent_title and \
                            cur_ent_wikiid != index_ent_wikiid:
                        num_nonexistent_both += 1
                    elif index_ent_title != cur_ent_title:
                        assert cur_ent_wikiid == index_ent_wikiid
                        num_nonexistent_ent_title += 1
                    else:
                        assert index_ent_title == cur_ent_title
                        assert cur_ent_wikiid != index_ent_wikiid
                        num_nonexistent_ent_id += 1

                    num_total_ents += 1  # Keep even incorrect links

                    cur_mentions.append({
                        'mention': cur_mention,
                        'ent_wikiid': final_ent_wikiid,
                        'start_off': len(cur_words) + 1,
                        'end_off': len(cur_words) + len(parts[2].split(' '))
                    })

                words = split_in_words(parts[0])
                cur_words += [modify_uppercase_phrase(w) for w in words]

        write_results(cur_doc_name, cur_mentions, cur_words,
                      yago_crosswikis_index, e_name_id, ouf)

        ouf.close()
        print('    Done AIDA.')
        print('num_nme = {}; num_nonexistent_ent_title = {}'
              .format(num_nme, num_nonexistent_ent_title))
        print('num_nonexistent_ent_id = {}; num_nonexistent_both = {}'
              .format(num_nonexistent_ent_id, num_nonexistent_both))
        print('num_correct_ents = {}; num_total_ents = {}'
              .format(num_correct_ents, num_total_ents))


def write_results(cur_doc_name, cur_mentions, cur_words,
                  yago_crosswikis_index, e_name_id, ouf):
    if not cur_doc_name or cur_doc_name == '':
        return
    header = cur_doc_name + '\t' + cur_doc_name + '\t'
    for hyp in cur_mentions:
        mention = hyp['mention']
        assert len(mention) > 0

        ctxt_str = header + mention + '\t'

        start = max(0, hyp['start_off'] - 101)
        end = hyp['start_off'] - 1
        left_ctxt = cur_words[start: end]
        if not left_ctxt:
            left_ctxt = ['EMPTYCTXT']
        if None in left_ctxt:
            print('None in cntxt of len: {} with doc name:{}, mention: {}'
                  .format(len(left_ctxt), cur_doc_name, hyp))
            print('cur words: {}'.format(cur_words))

        assert ' '.join(left_ctxt) + '\t'
        ctxt_str += ' '.join(left_ctxt) + '\t'

        start = hyp['end_off']
        end = min(len(cur_words), hyp['end_off'] + 100)
        right_ctxt = cur_words[start: end]
        if not right_ctxt:
            right_ctxt = ['EMPTYCTXT']

        ctxt_str += ' '.join(right_ctxt) + '\tCANDIDATES\t'

        # Entity candidates from p(e|m) dictionary
        if mention in yago_crosswikis_index.ent_p_e_m_index and \
                len(yago_crosswikis_index.ent_p_e_m_index[mention]) > 0:
            sorted_cand = {}
            for ent_wikiid in yago_crosswikis_index.ent_p_e_m_index[mention]:
                sorted_cand[ent_wikiid] = \
                    yago_crosswikis_index.ent_p_e_m_index[mention][ent_wikiid]

            sorted_cand = sorted(sorted_cand.items(), key=lambda kv: kv[1],
                                 reverse=True)
            gt_pos = -1
            pos = 0
            candidates = []
            for ent_wikiid, p in sorted_cand:
                if pos > 100:
                    break

                candidates.append(
                    '{},{:.3f},{}'.format(ent_wikiid, p, e_name_id
                                          .ent_from_wikiid(ent_wikiid)))

                pos += 1
                if ent_wikiid == hyp['ent_wikiid']:
                    gt_pos = pos

            ctxt_str += '\t'.join(candidates) + '\tGT:\t'

            if gt_pos > 0:
                ouf.write('{}{},{}\n'.format(ctxt_str, gt_pos,
                                             candidates[gt_pos - 1]))
            elif hyp['ent_wikiid'] != e_name_id.unk_ent_wikiid:
                ouf.write('{}-1,{},{}\n'
                          .format(ctxt_str, hyp['ent_wikiid'],
                                  e_name_id.ent_from_wikiid(hyp['ent_wikiid'])))
            else:
                ouf.write(ctxt_str + '-1\n')
        elif hyp['ent_wikiid'] != e_name_id.unk_ent_wikiid:
            ouf.write('{}EMPTYCAND\tGT:\t-1,{},{}\n'
                      .format(ctxt_str, hyp['ent_wikiid'],
                              e_name_id.ent_from_wikiid(hyp['ent_wikiid'])))
        else:
            ouf.write('{}EMPTYCAND\tGT:\t-1\n'.format(str))
