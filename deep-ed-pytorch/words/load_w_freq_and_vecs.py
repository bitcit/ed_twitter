# import os
# import torch
# from words.stop_words import is_stop_word_or_number
#
#
# Loads all common words in both Wikipedia and Word2vec/Glove , their unigram
# frequencies and their pre-trained Word2Vec embeddings.
# TODO: Filters words from word_wiki_count if they have embeddings.
#       Just complicates the code.
# def load_w_freq_and_vecs(args):
#     default_path = os.path.join(args.root_data_dir,
#                                 'basic_data/wordEmbeddings/')
#
#     torch.set_default_tensor_type(torch.FloatTensor)
#
#     assert args.word_vecs, 'Define args.word_vecs'
#     print('==> Loading common w2v + top freq list of words')
#
#     out = os.path.join(args.root_data_dir,
#                                'generated/common_top_words_freq_vectors_',
#                                args.word_vecs)
#     if os.path.isfile(out):
#         print('  ---> from dict file.')
#         common_w2v_freq_words = torch.load(out)
#     else:
#         print('  ---> dict file NOT found. Loading from disk instead (slower). '
#               'Out file = ' + out)
#         freq_words = {}
#
#         print('     word freq index ...')
#         w_freq_file = os.path.join(args.root_data_dir,
#                                    'generated/word_wiki_count.txt')
#         with open(w_freq_file, 'r') as f:
#             for line in f:
#                 line.rstrip()
#                 parts = line.split('\t')
#                 w = parts[0]
#                 w_f = int(parts[1])
#                 if not is_stop_word_or_number(w):
#                     freq_words[w] = w_f
#
#         common_w2v_freq_words = []
#
#         print('     word vectors index +.')
#         # if args.word_vecs == 'glove':
#             # Load glove, iterate through all words that have embeddings:
#             #     if w in freq_words:
#             #         common_w2v_freq_words.add(w)
#         # else:
#             # Load w2v, iterate through all words that have embeddings:
#             #     if w in freq_words:
#             #         common_w2v_freq_words.add(w)
#
#     print('Writing dict file for future usage. Next time it will be faster!')
#     torch.save(out, common_w2v_freq_words)
#     return common_w2v_freq_words
