# Definition of the neural network used to learn entity embeddings.

import torch
from torch import nn
from utils.utils import split_in_words, correct_type


class EntityEmbeddingsModel(nn.Module):
    def __init__(self, args, e_name_id, w2v, words, vecs_size=300):
        super(EntityEmbeddingsModel, self).__init__()
        self.args = args
        self.vecs_size = vecs_size
        self.e_name_id = e_name_id
        num_ents = e_name_id.get_total_num_ents()

        # Init ents vectors
        print('==> Init entity embeddings matrix. Num ents = {}'
              .format(num_ents))
        self.lookup_ent_vecs = nn.Embedding(num_ents, self.vecs_size)

        # Init entity vectors with average of title word embeddings.
        # This would help speed-up training.
        if self.args.init_vecs_title_words:
            print('Init entity embeddings with average of title word vectors.')
            init_ent_vec = correct_type(self.args,
                                        torch.zeros(num_ents, self.vecs_size))
            for ent_thid in range(1, num_ents):
                ent_name = e_name_id.get_ent_name_from_wikiid(
                    e_name_id.get_wikiid_from_thid(ent_thid))
                if not ent_name:
                    continue
                words_plus_stop_words = split_in_words(ent_name)

                num_words_title = 0
                for w in words_plus_stop_words:
                    if words.contains_w(w):
                        w_id = correct_type(self.args, torch.tensor(
                            [words.get_id_from_word(w)]))
                        e = w2v(w_id)
                        del w_id
                        init_ent_vec[ent_thid] = init_ent_vec[ent_thid].add(e)
                        del e
                        num_words_title += 1

                if num_words_title > 0:
                    if num_words_title > 3:
                        assert init_ent_vec[ent_thid].norm() > 0, ent_name
                    init_ent_vec[ent_thid] = \
                        init_ent_vec[ent_thid].div(num_words_title)

            self.lookup_ent_vecs.weight = nn.Parameter(init_ent_vec)
            del init_ent_vec

        print(' Done init.')

    def forward(self, input_layer):
        ctxt_w_vecs = input_layer[0][1]
        entity_thid_ids = input_layer[2][0]
        unig_distr_pow = input_layer[0][2]
        normalized_ctxt_vec = nn.functional.normalize(ctxt_w_vecs) \
            .reshape(self.args.batch_size,
                     self.args.num_words_per_ent * self.args.num_neg_words,
                     self.vecs_size)
        del ctxt_w_vecs

        entity_vecs = self.lookup_ent_vecs(entity_thid_ids)
        del entity_thid_ids

        normalized_ent_vecs = nn.functional.normalize(entity_vecs) \
            .reshape(self.args.batch_size, 1, self.vecs_size)
        del entity_vecs

        torch.cuda.synchronize()
        cosine_words_ents = torch.matmul(normalized_ctxt_vec,
                                         normalized_ent_vecs.permute(0, 2, 1)) \
            .reshape(self.args.batch_size * self.args.num_words_per_ent,
                     self.args.num_neg_words)
        del normalized_ctxt_vec
        del normalized_ent_vecs

        if self.args.loss == 'is':
            udl = torch.log(unig_distr_pow).reshape(self.args.batch_size *
                                                    self.args.num_words_per_ent,
                                                    self.args.num_neg_words)
            cosine_words_ents -= udl

        elif self.args.loss == 'nce':
            udlp = torch.log(unig_distr_pow * (self.args.num_neg_words - 1)) \
                .reshape(self.args.batch_size * self.args.num_words_per_ent,
                         self.args.num_neg_words)
            cosine_words_ents -= udlp

        return cosine_words_ents

    # ent id -> vec
    def geom_entwikiid2vec(self, ent_wikiid):
        ent_thid = self.e_name_id.get_thid(ent_wikiid)
        # invalid_ent_wikiids_stats(ent_thid)
        ev = self.lookup_ent_vecs.weight[ent_thid].float()
        ent_vec = nn.functional.normalize(ev, dim=0)
        return ent_vec

    def entity_similarity(self, e1_wikiid, e2_wikiid):
        e1_vec = self.geom_entwikiid2vec(e1_wikiid)
        e2_vec = self.geom_entwikiid2vec(e2_wikiid)
        return e1_vec.dot(e2_vec)
