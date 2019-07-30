import os
import torch
from math import pow, floor
from words.stop_words import is_stop_word_or_number


# Loads all words and their frequencies and IDs from a dictionary.
# assert common_w2v_freq_words
class Words:
    def __init__(self, args):
        print('==> Loading word freq map with unig power {}'
              .format(args.unig_power))
        w_freq_file = os.path.join(args.root_data_dir,
                                   'generated/word_wiki_count.txt')
        self.args = args
        self.id2word = {}
        self.word2id = {}

        self.wf_start = {}
        self.wf_end = {}
        self.total_freq = 0.0

        self.wf_unig_start = {}
        self.wf_unig_end = {}
        self.total_freq_unig = 0.0

        # UNK word id
        self.unk_w_id = 1
        self.word2id['UNK_W'] = self.unk_w_id
        self.id2word[self.unk_w_id] = 'UNK_W'

        w_id = 0
        with open(w_freq_file, 'r') as f:
            for line in f:
                line = line.rstrip()
                parts = line.split('\t')
                w = parts[0]
                if not is_stop_word_or_number(w):
                    w_id += 1
                    self.id2word[w_id] = w
                    self.word2id[w] = w_id

                    w_f = int(parts[1])
                    w_f = 100 if w_f < 100 else w_f

                    self.wf_start[w_id] = self.total_freq
                    self.total_freq = self.total_freq + w_f
                    self.wf_end[w_id] = self.total_freq

                    self.wf_unig_start[w_id] = self.total_freq_unig
                    self.total_freq_unig = \
                        self.total_freq_unig + pow(w_f, args.unig_power)
                    self.wf_unig_end[w_id] = self.total_freq_unig

        self.total_num_words = w_id

        print('  Done loading word freq index. Num words = {}; total freq = {}'
              .format(self.total_num_words, self.total_freq))

    def contains_w_id(self, w_id):
        assert 0 < w_id <= self.total_num_words, w_id
        return w_id != self.unk_w_id

    # id -> word
    def get_word_from_id(self, w_id):
        assert 0 < w_id <= self.total_num_words, w_id
        return self.id2word[w_id]

    # word -> id
    def get_id_from_word(self, w):
        return self.word2id[w] if w in self.word2id else self.unk_w_id

    def contains_w(self, w):
        return self.contains_w_id(self.get_id_from_word(w))

    # word frequency:
    def get_w_id_freq(self, w_id):
        # TODO:
        # assert self.contains_w_id(w_id), w_id
        return self.wf_end[w_id] - self.wf_start[w_id] + 1

    # p(w) prior:
    def get_w_id_unigram(self, w_id):
        return self.get_w_id_freq(w_id) / self.total_freq

    #
    # function get_w_tensor_log_unigram(vec_w_ids)
    #     assert(vec_w_ids:dim() == 2)
    #     v = torch.zeros(vec_w_ids:size(1), vec_w_ids:size(2))
    #     for i= 1,vec_w_ids:size(1):
    #         for j = 1,vec_w_ids:size(2):
    #             v[i][j] = math.log(get_w_id_unigram(vec_w_ids[i][j]))
    #         end
    #     end
    #     return v
    # end
    #
    #
    # if (args.unit_tests):
    #     print(get_w_id_unigram(get_id_from_word('the')))
    #     print(get_w_id_unigram(get_id_from_word('of')))
    #     print(get_w_id_unigram(get_id_from_word('and')))
    #     print(get_w_id_unigram(get_id_from_word('romania')))
    # end

    # Frequent word subsampling procedure from the Word2Vec paper.
    # Generates an random word sampled from the word unigram frequency.
    def random_unigram_at_unig_power_w_id(self):
        j = torch.rand(1) * self.total_freq_unig
        i_start = 1
        i_end = self.total_num_words

        while i_start <= i_end:
            i_mid = floor((i_start + i_end) / 2)
            w_id_mid = i_mid
            if self.wf_unig_start[w_id_mid] <= j <= self.wf_unig_end[w_id_mid]:
                return w_id_mid
            elif self.wf_unig_start[w_id_mid] > j:
                i_end = i_mid - 1
            elif self.wf_unig_end[w_id_mid] < j:
                i_start = i_mid + 1
        print('Binary search error !!')

    def get_w_unnorm_unigram_at_power(self, w_id):
        return pow(self.get_w_id_unigram(int(w_id)), self.args.unig_power)
#
#
# function unit_test_random_unigram_at_unig_power_w_id(k_samples)
#     empirical_dist = {}
#     for i=1,k_samples do
#         w_id = random_unigram_at_unig_power_w_id()
#         assert(w_id != self.unk_w_id)
#         if not empirical_dist[w_id]:
#             empirical_dist[w_id] = 0
#         end
#         empirical_dist[w_id] = empirical_dist[w_id] + 1
#     end
#     print('Now sorting +')
#     sorted_empirical_dist = {}
#     for k,v in pairs(empirical_dist):
#         table.insert(sorted_empirical_dist, {w_id = k, f = v})
#     end
#     table.sort(sorted_empirical_dist, function(a,b) return a.f > b.f end)
#
#     str = ''
#     for i = 1,math.min(100, table_len(sorted_empirical_dist)):
#         str = str + get_word_from_id(sorted_empirical_dist[i].w_id) + '{' + sorted_empirical_dist[i].f + '}; '
#     end
#     print('Unit test random sampling: ' + str)
# end
