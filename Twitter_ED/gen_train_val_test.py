import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-conll_file", type=str,
                    default='Microposts2014_train.conll')
parser.add_argument("-csv_file", type=str, default='train/micro_train.csv')
args = parser.parse_args()

doc_ids = set()
with open(args.csv_file, 'r', encoding='utf8') as f:
    for line in f:
        comps = line.strip().split('\t')
        doc_id = comps[0]
        assert int(doc_id), print("Wrong doc_id {} in csv".format(doc_id))
        doc_ids.add(doc_id)

out_file = args.csv_file.replace('.csv', '.conll')
ouf = open(out_file, 'w')
valid_example = False
with open(args.conll_file, 'r', encoding='utf8') as f:
    for line in f:
        line = line.strip()
        if line.startswith('-DOCSTART-'):
            doc_id = line.split()[1][1:]
            assert int(doc_id), print("Wrong doc_id {} in conll".format(doc_id))
            valid_example = doc_id in doc_ids
            if valid_example:
                ouf.write(line + '\n')
        elif valid_example:
            ouf.write(line + '\n')

ouf.close()
