#  Training of entity embeddings.

import argparse
import os
import time
import torch

from torch import nn
from entities.ent_name2id_freq.ent_name_id import EntityNameId
from entities.learn_e2v.model_a import EntityEmbeddingsModel
from words.w_freq.w_freq_index import Words
from words.w2v_utils import W2vUtils
from entities.learn_e2v.batch_dataset_a import EntityData
from entities.learn_e2v.minibatch_a import minibatch_to_correct_type

parser = argparse.ArgumentParser()
parser.add_argument('-root_data_dir', default='data/',
                    help='Root path of the data, $DATA_PATH.')
parser.add_argument('-type', default='cuda', help='Type: cpu | cuda ')
parser.add_argument('-optimizer', default='ADAGRAD',
                    help='RMSPROP | ADAGRAD | ADAM | SGD')
parser.add_argument('-lr', default=0.3, help='Learning rate.')
parser.add_argument('-batch_size', default=500, help='Mini-batch size.')
parser.add_argument('-word_vecs', default='w2v', help='glove | w2v')
parser.add_argument('-num_words_per_ent', default=20,
                    help='Number of positive words sampled for the given '
                         'entity at each iteration.')
parser.add_argument('-num_neg_words', default=5,
                    help='Number of negative words sampled for each positive '
                         'word.')
parser.add_argument('-unig_power', default=0.6,
                    help='Negative sampling unigram power (0.75 for Word2Vec).')
parser.add_argument('-entities', default='RLTD',
                    help='Entities for training embeddings: 4EX (for debug) | '
                         'RLTD (restricted set) | ALL (too big for one GPU)')
parser.add_argument('-init_vecs_title_words', default=True,
                    help='Init entity embeddings as the average of title words '
                         'embeddings (speeds up convergence).')
parser.add_argument('-loss', default='maxm',
                    help='nce (noise contrastive estimation) | '
                         'neg (negative sampling) | '
                         'is (importance sampling) | maxm (max-margin)')  # WTF?
parser.add_argument('-data', default='wiki-canonical-hyperlinks',
                    help='wiki-canonical (only) | wiki-canonical-hyperlinks')

# Only when args.data = wiki-canonical-hyperlinks
parser.add_argument('-num_passes_wiki_words', default=200,
                    help='Number of passes over Wiki canonical pages before '
                         'changing to using Wiki hyperlinks.')
parser.add_argument('-hyp_ctxt_len', default=10,
                    help='Left and right context window length for hyperlinks.')
parser.add_argument('-banner_header', default='', help='Banner header')
parser.add_argument('-wiki_redirects', default='basic_data/wiki_redirects.txt',
                    help='Wikipedia redirects index.')
parser.add_argument('-wiki_name_id_map',
                    default='basic_data/wiki_name_id_map.txt',
                    help='Wikipedia name id map.')
parser.add_argument('-wiki_disambiguation',
                    default='basic_data/wiki_disambiguation_pages.txt',
                    help='Wikipedia disambiguation index.')
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
parser.add_argument('-store_train_data', default='RAM',
                    help='Where to read the training data from: RAM (tensors) '
                         '| DISK (text, parsed all the time)')
parser.add_argument('-ctxt_window', default=100,
                    help='Number of context words at the left plus right of '
                         'each mention')
parser.add_argument('-num_cand_before_rerank', default=30)
args = parser.parse_args()

# Training of entity vectors
print('Learning entity vectors')

banner = '; '.join([a + '=' + str(getattr(args, a)) for a in vars(args)])

print('===> RUN TYPE: ' + args.type)

torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if 'cuda' in args.type and device != "cpu":
    print('==> switching to CUDA (GPU)')
else:
    print('==> running on CPU')

if args.loss == 'maxm':
    criterion = torch.nn.MultiMarginLoss(margin=0.1)
elif args.loss == 'is':
    criterion = torch.nn.CrossEntropyLoss()
else:
    criterion = torch.nn.SoftMarginLoss()  # TODO: or MultiLabelSoftMarginLoss?

e_name_id = EntityNameId(args)
words = Words(args)
w2vutils = W2vUtils(args)
model = EntityEmbeddingsModel(args, e_name_id, w2vutils.w2v, words)

if 'cuda' in args.type and device != "cpu":
    criterion = criterion.cuda()
    model.to(device)

if args.optimizer == 'ADAGRAD':
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)

elif args.optimizer == 'RMSPROP':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

print('Cuda alloc: {}, cached: {}'.format(torch.cuda.memory_allocated(),
                                              torch.cuda.memory_cached()))
print('Training entity vectors w/ params: ' + banner)

processed_so_far = 0
if args.entities == 'ALL':
    num_batches_per_epoch = 4000
elif args.entities == 'RLTD':
    num_batches_per_epoch = 2000
else:
    num_batches_per_epoch = 400

test_every_num_epochs = 1
save_every_num_epochs = 3
epochs = 85

train_data = EntityData(args, e_name_id, words, w2vutils)

# del words
# del w2vutils
torch.cuda.empty_cache()
print('Cuda alloc: {}, cached: {}'.format(torch.cuda.memory_allocated(),
                                              torch.cuda.memory_cached()))

for epoch in range(epochs):
    t0 = time.clock()
    print('\n===> TRAINING EPOCH #{}; num batches {} <==='
          .format(epoch, num_batches_per_epoch))

    avg_loss_before_args_per_epoch = 0.0
    avg_loss_after_args_per_epoch = 0.0

    loss = None
    loss_before = None
    sum_loss = 0
    for batch_index in range(num_batches_per_epoch):
        # Read one mini-batch from one data_thread:
        inputs, targets = train_data.get_minibatch()
        minibatch_to_correct_type(args, inputs)

        # optimize on current mini-batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if loss_before:
            if loss > loss_before:
                print('LOSS INCREASED: {} --> {}'.format(loss_before, loss))

        # Cast to float to prevent accumulating autograd history
        loss_before = float(loss.item())
        sum_loss += float(loss.item())

        # Display progress
        # train_size = 17000000  # 4 passes over the Wiki entity set
        # processed_so_far += args.batch_size
        # if processed_so_far > train_size:
        #     processed_so_far -= train_size
        # print('Processed so far: {}/{}'.format(processed_so_far, train_size))
    #   xlua.progress(processed_so_far, train_size)

    print('\nAvg loss = {}'.format(sum_loss / num_batches_per_epoch))

    # time taken
    duration = time.clock() - t0
    duration = duration / (num_batches_per_epoch * args.batch_size)
    print('==> time to learn 1 full entity = {} ms'.format(duration * 1000))

    # Various testing measures.
    if epoch % test_every_num_epochs == 0:
        if args.entities != '4EX' and e_name_id.rltd:
            e_name_id.rltd.compute_relatedness_metrics(model.entity_similarity)

    model_path = os.path.join(
        args.root_data_dir, 'generated/ent_vecs/ent_vecs__ep_{}'.format(epoch))
    # Save model.
    if epoch % save_every_num_epochs == 0:
        if epoch == epochs - 1:
            print('Normalizing entity vectors.')
            model.lookup_ent_vecs.weight.data = nn.functional.normalize(
                model.lookup_ent_vecs.weight.data)
        print('==> saving model to {}'.format(model_path))
        torch.save(model, model_path)
