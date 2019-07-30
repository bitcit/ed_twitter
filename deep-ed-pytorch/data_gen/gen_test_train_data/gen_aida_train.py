# Generate train data from the AIDA dataset by keeping the context and
# entity candidates for each annotated mention

# Format:
# doc_name \t doc_name \t mention \t left_ctxt \t right_ctxt \t CANDIDATES \t
# [ent_wikiid,p_e_m,ent_name]+ \t GT: \t pos,ent_wikiid,p_e_m,ent_name

from utils.utils import modify_uppercase_phrase, split_in_words


def gen_aida_train(aida_train, aida_train_out, yago_crosswikis_index,
                   entity_name_id):
    print('Generating train data from AIDA set ')
    ouf = open(aida_train_out, 'w')

    num_nme = 0
    num_nonexistent_ent_title = 0
    num_nonexistent_ent_id = 0
    num_nonexistent_both = 0
    num_correct_ents = 0
    num_total_ents = 0
    cur_words = []
    cur_mentions = []

    cur_doc_name = ''

    with open(aida_train, 'r') as f:
        for line in f:
            line = line.rstrip()

            if '-DOCSTART-' in line:
                write_results(cur_doc_name, cur_mentions, cur_words,
                              yago_crosswikis_index, entity_name_id, ouf)
                cur_doc_name = line[12:]

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
                    index_ent_title = entity_name_id.ent_from_wikiid(
                        cur_ent_wikiid)
                    index_ent_wikiid = entity_name_id.get_ent_wikiid_from_name(
                        cur_ent_title)

                    final_ent_wikiid = index_ent_wikiid
                    if final_ent_wikiid == entity_name_id.unk_ent_wikiid:
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
                      yago_crosswikis_index, entity_name_id, ouf)

        ouf.close()
        print('    Done AIDA.')
        print('num_nme = {}; num_nonexistent_ent_title = {}'
              .format(num_nme, num_nonexistent_ent_title))
        print('num_nonexistent_ent_id = {}; num_nonexistent_both = {}'
              .format(num_nonexistent_ent_id, num_nonexistent_both))
        print('num_correct_ents = {}; num_total_ents = {}'
              .format(num_correct_ents, num_total_ents))


def write_results(cur_doc_name, cur_mentions, cur_words,
                  yago_crosswikis_index, entity_name_id, ouf):
    if cur_doc_name == '':
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
            left_ctxt.append('EMPTYCTXT')

        ctxt_str += ' '.join(left_ctxt) + '\t'

        start = hyp['end_off']
        end = min(len(cur_words), hyp['end_off'] + 100)
        right_ctxt = cur_words[start: end]
        if not right_ctxt:
            right_ctxt.append('EMPTYCTXT')

        ctxt_str += ' '.join(right_ctxt) + '\tCANDIDATES\t'

        # Entity candidates from p(e|m) dictionary
        if mention not in yago_crosswikis_index.ent_p_e_m_index or \
                len(yago_crosswikis_index.ent_p_e_m_index[
                        mention]) < 1:
            continue

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

            candidates.append('{},{:.3f},{}'
                              .format(ent_wikiid, p, entity_name_id
                                      .ent_from_wikiid(ent_wikiid)))

            pos += 1
            if ent_wikiid == hyp['ent_wikiid']:
                gt_pos = pos

        ctxt_str += '\t'.join(candidates) + '\tGT:\t'

        if gt_pos > 0:
            ouf.write('{}{},{}\n'.format(ctxt_str, gt_pos,
                                         candidates[gt_pos - 1]))
