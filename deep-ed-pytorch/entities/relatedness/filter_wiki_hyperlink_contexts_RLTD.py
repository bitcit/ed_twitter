# Filter all training data s.t. only candidate entities and ground truth
# entities for which we have a valid entity embedding are kept.

# python3 -m entities.relatedness.filter_wiki_hyperlink_contexts_RLTD \
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
parser.add_argument('-wiki_ctxt',
                    default='generated/wiki_hyperlink_contexts.csv')
parser.add_argument('-wiki_ctxt_rltd',
                    default='generated/wiki_hyperlink_contexts_RLTD.csv')
args = parser.parse_args()

rltd = Relatedness(args)

ouf_path = os.path.join(args.root_data_dir, args.wiki_ctxt_rltd)
ouf = open(ouf_path, 'w')

num_lines = 0
print('Starting dataset filtering.')
wiki_context = os.path.join(args.root_data_dir, args.wiki_ctxt)
with open(wiki_context, 'r') as f:
    for line in f:
        line = line.rstrip()
        num_lines += 1
        if num_lines % 500000 == 0:
            print('Processed {} lines.'.format(num_lines))
        parts = line.split('\t')

        grd_str = parts[-1]
        assert parts[-2] == 'GT:', line

        grd_str_parts = grd_str.split(',')
        grd_pos = int(grd_str_parts[0])

        grd_ent_wikiid = int(grd_str_parts[1])
        assert grd_ent_wikiid

        if grd_ent_wikiid not in rltd.wikiid_to_rltdid:
            continue
        assert parts[5] == 'CANDIDATES'

        output_line = '\t'.join(parts[:6]) + '\t'

        new_grd_pos = -1
        new_grd_str_without_idx = None

        i = 1
        added_ents = 0
        while parts[5 + i] != 'GT:':
            hyp_str = parts[5 + i]
            str_parts = hyp_str.split(',')
            ent_wikiid = int(str_parts[0])
            if ent_wikiid in rltd.wikiid_to_rltdid:
                added_ents += 1
                output_line += hyp_str + '\t'

            if i == grd_pos:
                assert ent_wikiid == grd_ent_wikiid, 'Error for. ' + line
                new_grd_pos = added_ents
                new_grd_str_without_idx = hyp_str

            i += 1

        assert new_grd_pos > 0
        output_line = '{}GT.\t{},{}'.format(output_line, new_grd_pos,
                                            new_grd_str_without_idx)

        ouf.write(output_line + '\n')

ouf.close()
