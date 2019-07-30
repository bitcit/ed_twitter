# Loads pre-trained entity vectors trained by entity/learn_e2v/learn_a.lua

# from words.w2v_utils import W2vUtils


class E2vUtils:
    def __init__(self, args, ent_vecs_size=300):
        assert args.ent_vecs_filename
        print('==> Loading pre-trained entity vectors. e2v from file {}'
              .format(args.ent_vecs_filename))

        assert args.entities == 'RLTD', "Only RLTD entities are supported. " \
                                        "ALL entities can't fit to GPU memory."

        # geom_w2v_M = w2vutils.M.float()
        # w2v = W2vUtils(self.args)

        #
        # # Lookup table. ids -> tensor of vecs
        # e2vutils.lookup = torch.load(args.root_data_dir + 'generated/ent_vecs/' + args.ent_vecs_filename)
        # e2vutils.lookup = nn.Normalize(2).forward(e2vutils.lookup) # Needs to be normalized to have norm 1.
        #
        # assert e2vutils.lookup.size(1) == get_total_num_ents() and
        #     e2vutils.lookup.size(2) == ent_vecs_size, e2vutils.lookup.size(1) + ' ' + get_total_num_ents())
        # assert e2vutils.lookup[unk_ent_thid].norm() == 0, e2vutils.lookup[unk_ent_thid].norm())

    # ent wikiid -> vec
    # def entwikiid2vec (self, ent_wikiid, e_name_id):``
    #     thid = e_name_id.get_thid(ent_wikiid)
    #     return self.lookup[thid].float()

    # assert torch.norm(e2vutils.entwikiid2vec(unk_ent_wikiid)) == 0)
    #
    #
    # e2vutils.entname2vec = def (self,ent_name)
    #     assert ent_name)
    #     return e2vutils.entwikiid2vec(get_ent_wikiid_from_name(ent_name))
    # end
    #

    # Entity similarity based on cosine distance (vectors are normalized).
    # def entity_similarity(self, e1_wikiid, e2_wikiid):
    #     e1_vec = self.entwikiid2vec(e1_wikiid)
    #     e2_vec = self.entwikiid2vec(e2_wikiid)
    #     return e1_vec * e2_vec
#
#
# ###################################-
# ## Some unit tests to understand the quality of these embeddings ##
# ###################################-
# def geom_top_k_closest_words(ent_name, ent_vec, k)
#     tf_map = ent_wiki_words_4EX[ent_name]
#     w_not_found = {}
#     for w,_ in pairs(tf_map):
#         if tf_map[w] >= 10:
#             w_not_found[w] = tf_map[w]
#         end
#     end
#
#     distances = geom_w2v_M * ent_vec
#
#     best_scores, best_word_ids = topk(distances, k)
#     returnwords = {}
#     returndistances = {}
#     for i = 1,k:
#         w = get_word_from_id(best_word_ids[i])
#         if get_w_id_freq(best_word_ids[i]) >= 200:
#             w_freq_str = '[fr=' + get_w_id_freq(best_word_ids[i]) + ']'
#             if is_stop_word_or_number(w):
#                 table.insert(returnwords, red(w + w_freq_str))
#             elseif tf_map[w]:
#                 if tf_map[w] >= 15:
#                     table.insert(returnwords, yellow(w + w_freq_str + '{tf=' + tf_map[w] + '}'))
#                 else
#                     table.insert(returnwords, skyblue(w + w_freq_str + '{tf=' + tf_map[w] + '}'))
#                 end
#                 w_not_found[w] = nil
#             else
#                 table.insert(returnwords, w + w_freq_str)
#             end
#             assert best_scores[i] == distances[best_word_ids[i]], best_scores[i] + '    ' + distances[best_word_ids[i]])
#             table.insert(returndistances, distances[best_word_ids[i]])
#         end
#     end
#     return returnwords, returndistances, w_not_found
# end
#
#
# def geom_most_similar_words_to_ent(ent_name, k)
#     ent_wikiid = get_ent_wikiid_from_name(ent_name)
#     k = k or 1
#     ent_vec = e2vutils.entname2vec(ent_name)
#     assert math.abs(1 - ent_vec.norm()) < 0.01 or ent_vec.norm() == 0, '++ ' + ent_vec.norm())
#
#     print('\nTo entity. ' + blue(ent_name) + '; vec norm = ' + ent_vec.norm() + '.')
#     neighbors, scores, w_not_found = geom_top_k_closest_words(ent_name, ent_vec, k)
#     print(green('TOP CLOSEST WORDS. ') + list_with_scores_to_str(neighbors, scores))
#
#     str = yellow('WORDS NOT FOUND. ')
#     for w,tf in pairs(w_not_found):
#         if tf >= 20:
#             str = str + yellow(w + '{' + tf + '}; ')
#         else
#             str = str + w + '{' + tf + '}; '
#         end
#     end
#     print('\n' + str)
#     print('============================================================================')
# end
#
# def geom_unit_tests()
#     print('\n' + yellow('TOP CLOSEST WORDS to a given entity based on cosine distance.'))
#     print('For each word, we show the unigram frequency [fr] and the cosine similarity.')
#     print('Infrequent words [fr < 500] have noisy embeddings, thus should be trusted less.')
#     print('WORDS NOT FOUND contains frequent words from the Wikipedia canonical page that are not found in the TOP CLOSEST WORDS list.')
#
#     for i=1,table_len(ent_names_4EX):
#         geom_most_similar_words_to_ent(ent_names_4EX[i], 300)
#     end
# end
#
# print('        Done reading e2v data. Entity vocab size = ' + e2vutils.lookup.size(1))
