import os
import torch
from entities.learn_e2v.minibatch_a import empty_minibatch, process_one_line, \
    postprocess_minibatch
from utils.utils import correct_type


class EntityData:
    def __init__(self, args, e_name_id, words, w2vutils):
        self.args = args
        self.e_name_id = e_name_id
        self.words = words
        self.w2vutils = w2vutils
        if args.entities == 'ALL':
            self.wiki_words_train_file = \
                os.path.join(args.root_data_dir,
                             'generated/wiki_canonical_words.txt')
            self.wiki_hyp_train_file = \
                os.path.join(args.root_data_dir,
                             'generated/wiki_hyperlink_contexts.csv')
        else:
            self.wiki_words_train_file = \
                os.path.join(args.root_data_dir,
                             'generated/wiki_canonical_words_RLTD.txt')
            self.wiki_hyp_train_file = \
                os.path.join(args.root_data_dir,
                             'generated/wiki_hyperlink_contexts_RLTD.csv')

        self.wiki_words_f = open(self.wiki_words_train_file, 'r')
        self.wiki_hyp_f = open(self.wiki_hyp_train_file, 'r')

        self.train_data_source = 'wiki-canonical'
        self.num_passes_wiki_words = 1

    def read_one_line(self):
        if self.train_data_source == 'wiki-canonical':
            line = self.wiki_words_f.readline().rstrip()
        else:
            line = self.wiki_hyp_f.readline().rstrip()

        if line:
            return line

        if self.num_passes_wiki_words == self.args.num_passes_wiki_words:
            self.train_data_source = 'wiki-canonical-hyperlinks'
            print('\nStart training on Wiki Hyperlinks\n')

        print('Training file is done. Num passes = {}. Reopening.'
              .format(self.num_passes_wiki_words))
        self.num_passes_wiki_words += 1

        if self.train_data_source == 'wiki-canonical':
            self.wiki_words_f.close()
            self.wiki_words_f = open(self.wiki_words_train_file, 'r')
            line = self.wiki_words_f.readline().rstrip()
        else:
            self.wiki_hyp_f.close()
            self.wiki_hyp_f = open(self.wiki_hyp_train_file, 'r')
            line = self.wiki_hyp_f.readline().rstrip()

        return line

    def get_minibatch(self):
        # Create empty mini batch:
        inputs = empty_minibatch(self.args, self.words)
        targets = torch.ones(self.args.batch_size,
                             self.args.num_words_per_ent).long()

        # Fill in each example:
        for i in range(self.args.batch_size):
            line = self.read_one_line()
            inputs, target = process_one_line(self.args, line, inputs, i,
                                              self.e_name_id, self.words)
            targets[i] = target

        # Minibatch post processing:
        inputs = postprocess_minibatch(self.args, inputs, self.w2vutils,
                                       self.words)
        targets = targets.view(
            self.args.batch_size * self.args.num_words_per_ent)

        # Special target for the NEG and NCE losses
        if self.args.loss == 'neg' or self.args.loss == 'nce':
            nce_targets = torch.ones(
                self.args.batch_size * self.args.num_words_per_ent,
                self.args.num_neg_words).mul(-1).long()
            for j in range(self.args.batch_size * self.args.num_words_per_ent):
                nce_targets[j][targets[j]] = 1

            targets = nce_targets

        return inputs, correct_type(self.args, targets)
