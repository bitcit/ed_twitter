import os
import torch

from gensim.models import KeyedVectors


class W2vUtils:
    def __init__(self, args, dim=300):
        # Loads pre-trained glove or word2vec embeddings:
        self.args = args
        default_path = os.path.join(args.root_data_dir,
                                    'basic_data/wordEmbeddings/')
        if args.word_vecs == 'glove':
            # Glove from: http://nlp.stanford.edu/projects/glove/
            w2v_pretrained = os.path.join(default_path,
                                          'glove/glove.6B.50d.txt')
            w2v_dict = os.path.join(args.root_data_dir,
                                    'generated/glove.6B.50d')
            binary = False
        else:
            # Word2Vec from: https://code.google.com/archive/p/word2vec/
            w2v_pretrained = os.path.join(
                default_path, 'Word2Vec/GoogleNews-vectors-negative300.bin')
            w2v_dict = os.path.join(args.root_data_dir,
                                    'generated/GoogleNews-vectors-negative300')
            binary = True

        print('==> Loading {} vectors'.format(args.word_vecs))
        if not os.path.isfile(w2v_dict):
            print(
                '  ---> dict file NOT found. Loading w2v from the bin/txt file '
                'instead (slower).')
            gensim_w2v = KeyedVectors.load_word2vec_format(w2v_pretrained,
                                                           binary=binary)
            weights = torch.FloatTensor(gensim_w2v.syn0)
            self.w2v = torch.nn.Embedding.from_pretrained(weights)
            print('Writing dict File for future usage. Next time Word2Vec '
                  'loading will be faster!')
            torch.save(self.w2v, w2v_dict)
        else:
            print('  ---> from dict file.')
            self.w2v = torch.load(w2v_dict)

        # Move the word embedding matrix on the GPU if we do some training.
        # In this way we can perform word embedding lookup much faster.
        if 'cuda' in args.type:
            self.w2v = self.w2v.cuda()

    # # word -> vec
    # w2v.get_w_vec = function (self,word)
    #   w_id = get_id_from_word(word)
    #   return w2v.M[w_id].clone()
    # end
    #
    # # word_id -> vec
    # w2v.get_w_vec_from_id = function (self,w_id)
    #   return w2v.M[w_id].clone()
    # end
    #
    def lookup_w_vecs(self, word_id_tensor):
        assert word_id_tensor.dim() <= 2, 'Only word id tensors w/ 1 or 2 ' \
                                          'dimensions are supported.'
        word_ids = word_id_tensor.long()
        if self.args and 'cuda' in self.args.type:
            word_ids = word_ids.cuda()

        if word_ids.dim() == 2:
            output = self.w2v(word_ids.view(-1))
            output = output.view(word_ids.size(0), word_ids.size(1),
                                 self.w2v.weight.size(1))
        elif word_ids.dim() == 1:
            output = self.w2v(word_ids)
            output = output.view(word_ids.size(0), self.w2v.weight.size(1))

        if self.args and 'cuda' in self.args.type:
            output = output.cuda()

        return output
    #
    # # Normalize word vectors to have norm 1 .
    # w2v.renormalize = function (self)
    #   w2v.M[unk_w_id]:mul(0)
    #   w2v.M[unk_w_id]:add(1)
    #   w2v.M.cdiv(w2v.M:norm(2,2):expand(w2v.M:size()))
    #   x = w2v.M:norm(2,2):view(-1) - 1
    #   assert(x:norm() < 0.1, x:norm())
    #   assert(w2v.M[100]:norm() < 1.001 and w2v.M[100]:norm() > 0.99)
    #   w2v.M[unk_w_id]:mul(0)
    # end
    #
    # w2v:renormalize()
    #
    # print('    Done reading w2v data. Word vocab size = ' + w2v.M:size(1))
    #
    # # Phrase embedding using average of vectors of words in the phrase
    # w2v.phrase_avg_vec = function(self, phrase)
    #   words = split_in_words(phrase)
    #   num_words = table_len(words)
    #   num_existent_words = 0
    #   vec = torch.zeros(dim)
    #   for i = 1,num_words:
    #     w = words[i]
    #     w_id = get_id_from_word(w)
    #     if w_id ~= unk_w_id:
    #       vec:add(w2v:get_w_vec_from_id(w_id))
    #       num_existent_words = num_existent_words + 1
    #     end
    #   end
    #   if (num_existent_words > 0):
    #     vec.div(num_existent_words)
    #   end
    #   return vec
    # end
    #
    # w2v.top_k_closest_words = function (self,vec, k, mat)
    #   k = k or 1
    #   vec = vec:float()
    #   distances = torch.mv(mat, vec)
    #   best_scores, best_word_ids = topk(distances, k)
    #   returnwords = {}
    #   returndistances = {}
    #   for i = 1,k:
    #     w = get_word_from_id(best_word_ids[i])
    #     if is_stop_word_or_number(w):
    #       table.insert(returnwords, red(w))
    #     else:
    #       table.insert(returnwords, w)
    #     end
    #     assert(best_scores[i] == distances[best_word_ids[i]], best_scores[i] + '  ' + distances[best_word_ids[i]])
    #     table.insert(returndistances, distances[best_word_ids[i]])
    #   end
    #   return returnwords, returndistances
    # end
    #
    # w2v.most_similar2word = function(self, word, k)
    #   k = k or 1
    #   v = w2v:get_w_vec(word)
    #   neighbors, scores = w2v:top_k_closest_words(v, k, w2v.M)
    #   print('To word ' + skyblue(word) + ' : ' + list_with_scores_to_str(neighbors, scores))
    # end
    #
    # w2v.most_similar2vec = function(self, vec, k)
    #   k = k or 1
    #   neighbors, scores = w2v:top_k_closest_words(vec, k, w2v.M)
    #   print(list_with_scores_to_str(neighbors, scores))
    # end
    #
    #
    # ##########- Unit tests ####################
    # unit_tests = args.unit_tests or false
    # if (unit_tests):
    #   print('\nWord to word similarity test:')
    #   w2v:most_similar2word('nice', 5)
    #   w2v:most_similar2word('france', 5)
    #   w2v:most_similar2word('hello', 5)
    # end
    #
    # # Computes for each word w : \sum_v exp(<v,w>) and \sum_v <v,w>
    # w2v.total_word_correlation = function(self, k, j)
    #   exp_Z = torch.zeros(w2v.M:narrow(1, 1, j):size(1))
    #
    #   sum_t = w2v.M:narrow(1, 1, j):sum(1) # 1 x d
    #   sum_Z = (w2v.M:narrow(1, 1, j) * sum_t:t()):view(-1) # num_w
    #
    #   print(red('Top words by sum_Z:'))
    #   best_sum_Z, best_word_ids = topk(sum_Z, k)
    #   for i = 1,k:
    #     w = get_word_from_id(best_word_ids[i])
    #     assert(best_sum_Z[i] == sum_Z[best_word_ids[i]])
    #     print(w + ' [' + best_sum_Z[i] + ']; ')
    #   end
    #
    #   print('\n' + red('Bottom words by sum_Z:'))
    #   best_sum_Z, best_word_ids = topk(- sum_Z, k)
    #   for i = 1,k:
    #     w = get_word_from_id(best_word_ids[i])
    #     assert(best_sum_Z[i] == - sum_Z[best_word_ids[i]])
    #     print(w + ' [' + sum_Z[best_word_ids[i]] + ']; ')
    #   end
    # end
    #
    #
    # # Plot with gnuplot:
    # # set palette model RGB defined ( 0 'white', 1 'pink', 2 'green' , 3 'blue', 4 'red' )
    # # plot 'tsne-w2v-vecs.txt_1000' using 1:2:3 with labels offset 0,1, '' using 1:2:4 w points pt 7 ps 2 palette
    # w2v.tsne = function(self, num_rand_words)
    #   topic1 = {'japan', 'china', 'france', 'switzerland', 'romania', 'india', 'australia', 'country', 'city', 'tokyo', 'nation', 'capital', 'continent', 'europe', 'asia', 'earth', 'america'}
    #   topic2 = {'football', 'striker', 'goalkeeper', 'basketball', 'coach', 'championship', 'cup',
    #     'soccer', 'player', 'captain', 'qualifier', 'goal', 'under-21', 'halftime', 'standings', 'basketball',
    #     'games', 'league', 'rugby', 'hockey', 'fifa', 'fans', 'maradona', 'mutu', 'hagi', 'beckham', 'injury', 'game',
    #     'kick', 'penalty'}
    #   topic_avg = {'japan national football team', 'germany national football team',
    #     'china national football team', 'brazil soccer', 'japan soccer', 'germany soccer', 'china soccer',
    #     'fc barcelona', 'real madrid'}
    #
    #   stop_words_array = {}
    #   for w,_ in pairs(stop_words):
    #     table.insert(stop_words_array, w)
    #   end
    #
    #   topic1_len = table_len(topic1)
    #   topic2_len = table_len(topic2)
    #   topic_avg_len = table_len(topic_avg)
    #   stop_words_len = table_len(stop_words_array)
    #
    #   torch.setdefaulttensortype('torch.DoubleTensor')
    #   w2v.M = w2v.M.double()
    #
    #   tensor = torch.zeros(num_rand_words + stop_words_len + topic1_len + topic2_len + topic_avg_len, dim)
    #   tensor_w_ids = torch.zeros(num_rand_words)
    #   tensor_colors = torch.zeros(tensor:size(1))
    #
    #   for i = 1,num_rand_words:
    #     tensor_w_ids[i] = math.random(1,25000)
    #     tensor_colors[i] = 0
    #     tensor[i].copy(w2v.M[tensor_w_ids[i]])
    #   end
    #
    #   for i = 1, stop_words_len:
    #     tensor_colors[num_rand_words + i] = 1
    #     tensor[num_rand_words + i].copy(w2v:phrase_avg_vec(stop_words_array[i]))
    #   end
    #
    #   for i = 1, topic1_len:
    #     tensor_colors[num_rand_words + stop_words_len + i] = 2
    #     tensor[num_rand_words + stop_words_len + i].copy(w2v:phrase_avg_vec(topic1[i]))
    #   end
    #
    #   for i = 1, topic2_len:
    #     tensor_colors[num_rand_words + stop_words_len + topic1_len + i] = 3
    #     tensor[num_rand_words + stop_words_len + topic1_len + i].copy(w2v:phrase_avg_vec(topic2[i]))
    #   end
    #
    #   for i = 1, topic_avg_len:
    #     tensor_colors[num_rand_words + stop_words_len + topic1_len  + topic2_len + i] = 4
    #     tensor[num_rand_words + stop_words_len + topic1_len  + topic2_len + i].copy(w2v:phrase_avg_vec(topic_avg[i]))
    #   end
    #
    #   manifold = require 'manifold'
    #   argss = {ndims = 2, perplexity = 30, pca = 50, use_bh = false}
    #   mapped_x1 = manifold.embedding.tsne(tensor, argss)
    #   assert(mapped_x1:size(1) == tensor:size(1) and mapped_x1:size(2) == 2)
    #   ouf_vecs = assert(io.open('tsne-w2v-vecs.txt_' + num_rand_words, "w"))
    #   for i = 1,mapped_x1:size(1):
    #     w = nil
    #     if tensor_colors[i] == 0:
    #       w = get_word_from_id(tensor_w_ids[i])
    #     elif tensor_colors[i] == 1:
    #       w = stop_words_array[i - num_rand_words]:gsub(' ', '-')
    #     elif tensor_colors[i] == 2:
    #       w = topic1[i - num_rand_words - stop_words_len]:gsub(' ', '-')
    #     elif tensor_colors[i] == 3:
    #       w = topic2[i - num_rand_words - stop_words_len - topic1_len]:gsub(' ', '-')
    #     elif tensor_colors[i] == 4:
    #       w = topic_avg[i - num_rand_words - stop_words_len - topic1_len - topic2_len]:gsub(' ', '-')
    #     end
    #     assert(w)
    #
    #     v = mapped_x1[i]
    #     for j = 1,2:
    #       ouf_vecs:write(v[j] + ' ')
    #     end
    #     ouf_vecs:write(w + ' ' + tensor_colors[i] + '\n')
    #   end
    #   io.close(ouf_vecs)
    #   print('    DONE')
    # end
