# Computes an unigram frequency of each word in the Wikipedia corpus

# python3 -m words.w_freq.w_freq_gen \
# -root_data_dir small-data/

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-root_data_dir', default='data/',
                    help='Root path of the data, $DATA_PATH.')
parser.add_argument('-wiki_can_words',
                    default='generated/wiki_canonical_words.txt',
                    help='Words from canonical wikipedia pages.')
parser.add_argument('-wiki_word_count',
                    default='generated/word_wiki_count.txt')
args = parser.parse_args()

word_counts = {}
num_lines = 0

wiki_canonical_words = os.path.join(args.root_data_dir, args.wiki_can_words)
with open(wiki_canonical_words, 'r') as f:
    for line in f:
        line = line.rstrip()
        num_lines += 1
        if num_lines % 500000 == 0:
            print('Processed {} lines.'.format(num_lines))
        parts = line.split('\t')
        words = parts[2].split(' ')
        for w in words:
            if w not in word_counts:
                word_counts[w] = 0

            word_counts[w] += 1

print('Sorting and writing')

filtered_wc = [(k, v) for k, v in word_counts.items() if v >= 10]
sorted_wc = sorted(filtered_wc, key=lambda kv: kv[1], reverse=True)

total_count = 0

out_file = os.path.join(args.root_data_dir, args.wiki_word_count)
with open(out_file, 'w') as f:
    for w, cnt in sorted_wc:
        f.write('{}\t{}\n'
                .format(w, cnt))
        total_count += cnt

print('Total count = {}'.format(total_count))
