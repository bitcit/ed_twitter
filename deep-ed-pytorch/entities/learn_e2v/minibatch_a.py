import torch
from utils.utils import split_in_words, correct_type

unk_w_id = 1


def empty_minibatch(args, words):
    ctxt_word_ids = torch.ones(args.batch_size,
                               args.num_words_per_ent,
                               args.num_neg_words)
    # Sample negative words from \hat{p}(w)^\alpha.
    ctxt_word_ids.apply_(lambda x: words.random_unigram_at_unig_power_w_id())
    ent_component_words = torch.ones(args.batch_size,
                                     args.num_words_per_ent).int()
    ent_thids = torch.ones(args.batch_size).long()
    ent_wikiids = torch.ones(args.batch_size).int()
    return [[ctxt_word_ids, 0, 0], [ent_component_words],
            [ent_thids, ent_wikiids]]


# Get defs:
def get_pos_and_neg_w_ids(minibatch):
    return minibatch[0][0]


def get_pos_and_neg_w_vecs(minibatch):
    return minibatch[0][1]


def get_pos_and_neg_w_unig_at_power(minibatch):
    return minibatch[0][2]


def get_ent_wiki_w_ids(minibatch):
    return minibatch[1][0]


# def get_ent_wiki_w_vecs(minibatch):
#     return minibatch[1][1]


def get_ent_thids_batch(minibatch):
    return minibatch[2][0]


def get_ent_wikiids(minibatch):
    return minibatch[2][1]


# Fills in the minibatch and returns the grd truth word index per each example.
# An example in our case is an entity, a positive word sampled from \hat{p}(e|m)
# and several negative words sampled from \hat{p}(w)^\alpha.
def process_one_line(args, line, minibatch, mb_index, e_name_id, words):
    # if args.entities == '4EX':
    #     line = ent_lines_4EX[ent_names_4EX[torch.randint(len(ent_names_4EX))]]

    parts = line.split('\t')
    if len(parts) == 3:  # Words from the Wikipedia canonical page
        ent_wikiid = int(parts[0])
        words_plus_stop_words = parts[2].split(' ')

    else:  # Words from Wikipedia hyperlinks
        assert len(parts) >= 9, line + ' -> #' + len(parts)
        assert parts[5] == 'CANDIDATES', line

        ent_str = parts[-1].split(',')
        ent_wikiid = int(ent_str[1])

        left_ctxt_w = parts[3].split(' ')
        start = max(0, len(left_ctxt_w) - args.hyp_ctxt_len)
        words_plus_stop_words = left_ctxt_w[start:]

        right_ctxt_w = parts[4].split(' ')
        end = min(len(right_ctxt_w), args.hyp_ctxt_len)
        words_plus_stop_words += right_ctxt_w[:end]

    ent_thid = e_name_id.get_thid(ent_wikiid)
    assert e_name_id.get_wikiid_from_thid(ent_thid) == ent_wikiid
    get_ent_thids_batch(minibatch)[mb_index] = ent_thid
    assert get_ent_thids_batch(minibatch)[mb_index] == ent_thid

    get_ent_wikiids(minibatch)[mb_index] = ent_wikiid

    # Remove stop words from entity wiki words representations.
    pos_w = [w for w in words_plus_stop_words if words.contains_w(w)]

    # Get some words from the entity title if the canonical page is empty.
    if not pos_w:
        ent_name = parts[1]
        pos_w = [w for w in split_in_words(ent_name) if words.contains_w(w)]

        # Still empty ? Get some random words:.
        if not pos_w:
            pos_w = words.get_word_from_id(
                words.random_unigram_at_unig_power_w_id())

    targets = torch.zeros(args.num_words_per_ent)

    # Sample some positive words:
    for i in range(args.num_words_per_ent):
        positive_w = pos_w[torch.randint(len(pos_w), (1,))]
        positive_w_id = words.get_id_from_word(positive_w)

        # Set the positive word in a random position.
        # Remember that index (used in training).
        grd_trth = torch.randint(args.num_neg_words, (1,))
        get_ent_wiki_w_ids(minibatch)[mb_index][i] = positive_w_id
        assert get_ent_wiki_w_ids(minibatch)[mb_index][i] == positive_w_id
        targets[i] = grd_trth
        get_pos_and_neg_w_ids(minibatch)[mb_index][i][grd_trth] = positive_w_id

    return minibatch, targets


# Fill minibatch with word and entity vectors:
def postprocess_minibatch(args, minibatch, w2vutils, words):
    minibatch[0][0] = get_pos_and_neg_w_ids(minibatch) \
        .view(args.batch_size * args.num_words_per_ent * args.num_neg_words)
    minibatch[1][0] = get_ent_wiki_w_ids(minibatch).view(
        args.batch_size * args.num_words_per_ent)

    # ctxt word vecs
    minibatch[0][1] = w2vutils.lookup_w_vecs(get_pos_and_neg_w_ids(minibatch))
    u = [words.get_w_unnorm_unigram_at_power(w_id) for w_id in minibatch[0][0]]
    minibatch[0][2] = torch.tensor(u)
    return minibatch


# Convert mini batch to correct type (e.g. move data to GPU):
def minibatch_to_correct_type(args, minibatch):
    minibatch[0][0] = correct_type(args, minibatch[0][0])
    minibatch[1][0] = correct_type(args, minibatch[1][0])
    minibatch[0][1] = correct_type(args, minibatch[0][1])
    minibatch[0][2] = correct_type(args, minibatch[0][2])
    minibatch[2][0] = correct_type(args, minibatch[2][0])
    return minibatch
