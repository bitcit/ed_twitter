# The code in this file does two things:
#   a) extracts and puts the entity relatedness dataset in two maps (reltd
#      validate and reltd_test). Provides functions to evaluate entity
#      embeddings on this dataset (Table 1 in our paper).
#   b) extracts all entities that appear in any of the ED (as mention
#      candidates) or entity relatedness datasets. These are placed in
#      Relatedness class that will be used to restrict the set of entities
#      for which we want to train entity embeddings (done with the file
#      entities/learn_e2v/learn_a.lua).

import os.path
import torch

from math import log


class Relatedness:
    def __init__(self, args):
        rel_test_txt = os.path.join(args.root_data_dir, args.rltd_test_txt)
        rel_val_txt = os.path.join(args.root_data_dir, args.rltd_val_txt)

        rel_test_dict = os.path.join(args.root_data_dir, args.rltd_test_dict)
        rel_val_dict = os.path.join(args.root_data_dir, args.rltd_val_dict)

        self.reltd_validate = load_reltd_set(rel_val_dict, rel_val_txt,
                                             'validate')
        self.reltd_test = load_reltd_set(rel_test_dict, rel_test_txt, 'test')

        reltd_ents_direct_validate = extract_reltd_ents(self.reltd_validate)
        reltd_ents_direct_test = extract_reltd_ents(self.reltd_test)

        rltd_dict_path = os.path.join(args.root_data_dir, args.rltd_dict)
        print('==> Loading relatedness thid tensor')
        if not os.path.exists(rltd_dict_path):
            print('  ---> dict file NOT found. Loading reltd_ents_wikiid_to_'
                  'rltdid from txt file instead (slower).')

            # Restricted set of entities for which we train entity embeddings:
            rltd_all_ent_wikiids = set()

            # 1) From the relatedness dataset
            for ent_wikiid in reltd_ents_direct_validate:
                rltd_all_ent_wikiids.add(ent_wikiid)
            for ent_wikiid in reltd_ents_direct_test:
                rltd_all_ent_wikiids.add(ent_wikiid)

            # TODO: add
            # 1.1) From a small dataset (used for debugging / unit testing).
            # for line in ent_lines_4EX):
            #     parts = line.split('\t')
            #     assert(len(parts) == 3)
            #     ent_wikiid = int(parts[0])
            #     # assert(ent_wikiid)
            #     rltd_all_ent_wikiids[ent_wikiid] = 1

            # TODO: put to args
            # 2) From all ED datasets:
            files = {'aida_train.csv', 'aida_testA.csv', 'aida_testB.csv',
                     'wned-aquaint.csv', 'wned-msnbc.csv', 'wned-ace2004.csv',
                     'wned-clueweb.csv', 'wned-wikipedia.csv'}

            data_path = os.path.join(args.root_data_dir, args.test_train)

            for file in files:
                with open(os.path.join(data_path, file), 'r') as f:
                    for line in f:
                        line = line.rstrip()
                        if not line or 'EMPTYCAND' in line:
                            continue
                        parts = line.split('\t')
                        assert parts[5] == 'CANDIDATES', line
                        assert parts[-2] == 'GT:', line

                        for part in parts[6:-2]:
                            p = part.split(',')
                            ent_wikiid = int(p[0])
                            assert ent_wikiid
                            rltd_all_ent_wikiids.add(ent_wikiid)

            # Insert unk_ent_wikiid
            unk_ent_wikiid = 1
            rltd_all_ent_wikiids.add(unk_ent_wikiid)

            # Sort all wikiids
            sorted_rltd_all_ent_wikiids = sorted(rltd_all_ent_wikiids)

            if not sorted_rltd_all_ent_wikiids:
                sorted_rltd_all_ent_wikiids = []

            reltd_ents_wikiid_to_rltdid = {}
            for rltd_id, wikiid in enumerate(sorted_rltd_all_ent_wikiids):
                reltd_ents_wikiid_to_rltdid[wikiid] = rltd_id

            self.wikiid_to_rltdid = reltd_ents_wikiid_to_rltdid
            self.rltdid_to_wikiid = sorted_rltd_all_ent_wikiids

            print('Writing reltd_ents_wikiid_to_rltdid to dict File for future '
                  'usage.')
            torch.save(self, rltd_dict_path)
            print('    Done saving.')
        else:
            print('  ---> from dict file.')
            other = torch.load(rltd_dict_path)
            self.wikiid_to_rltdid = other.wikiid_to_rltdid
            self.rltdid_to_wikiid = other.rltdid_to_wikiid

        print('    Done loading relatedness sets. Num queries test = {}. '
              'Num queries valid = {}. Total num ents restricted set = {}'
              .format(len(self.reltd_test), len(self.reltd_validate),
                      len(self.rltdid_to_wikiid)))

    # Main function that computes results for the entity relatedness dataset
    # (Table 1 of the paper) given any entity similarity function as input.
    def compute_relatedness_metrics(self, entity_sim):
        print('Entity Relatedness quality measure:')
        ideals_rltd_validate_scores = compute_ideal_rltd_scores(
            self.reltd_validate)
        ideals_rltd_test_scores = compute_ideal_rltd_scores(self.reltd_test)

        assert abs(compute_map(ideals_rltd_validate_scores,
                               self.reltd_validate) - 1) < 0.001
        assert abs(
            compute_map(ideals_rltd_test_scores, self.reltd_test) - 1) < 0.001

        scores_validate = compute_e2v_rltd_scores(self.reltd_validate,
                                                  entity_sim)
        scores_test = compute_e2v_rltd_scores(self.reltd_test, entity_sim)

        map_val = compute_map(scores_validate, self.reltd_validate)
        ndcg_1_val = compute_NDCG(1, scores_validate, self.reltd_validate,
                                  ideals_rltd_validate_scores)
        ndcg_5_val = compute_NDCG(5, scores_validate, self.reltd_validate,
                                  ideals_rltd_validate_scores)
        ndcg_10_val = compute_NDCG(10, scores_validate, self.reltd_validate,
                                   ideals_rltd_validate_scores)
        total = map_val + ndcg_1_val + ndcg_5_val + ndcg_10_val

        map_test = compute_map(scores_test, self.reltd_test)
        ndcg_1_test = compute_NDCG(1, scores_test, self.reltd_test,
                                   ideals_rltd_test_scores)
        ndcg_5_test = compute_NDCG(5, scores_test, self.reltd_test,
                                   ideals_rltd_test_scores)
        ndcg_10_test = compute_NDCG(10, scores_test, self.reltd_test,
                                    ideals_rltd_test_scores)

        print('measure    =   NDCG1   NDCG5   NDCG10  MAP   TOTAL VALIDATION')
        print('our (vald) =\t{}\t{}\t{}\t{}\t{}'
              .format(ndcg_1_val, ndcg_5_val, ndcg_10_val, map_val, total))
        print('our (vald) =\t{}\t{}\t{}\t{}'
              .format(ndcg_1_test, ndcg_5_test, ndcg_10_test, map_test))
        print('Yamada\'16  =    0.59    0.56    0.59    0.52')
        print('WikiMW     =    0.54    0.52    0.55    0.48')


# Loads the entity relatedness dataset (validation and test parts) to a map
# called reltd.
# Format: reltd = {query_id q -> (query_entity e1, entity_candidates cand) }
#         cand = {e2 -> label}, where label is binary, if the candidate entity
#                               is related to e1
def load_reltd_set(rel_dict, rel_txt, set_type):
    print('==> Loading relatedness ' + set_type)
    if os.path.isfile(rel_dict):
        print('  ---> from dict file.')
        return torch.load(rel_dict)

    print('  ---> dict file NOT found. Loading relatedness {} from txt file '
          'instead (slower).'.format(set_type))
    reltd = {}
    with open(rel_txt, 'r') as f:
        for line in f:
            line = line.rstrip()
            parts = line.split(' ')
            label = int(parts[0])

            assert label == 0 or label == 1

            t = parts[1].split(':')
            q = int(t[1])
            i = parts.index('#')

            ents = parts[i + 1].split('-')
            e1 = int(ents[0])
            e2 = int(ents[1])

            if q not in reltd:
                reltd[q] = {}
                reltd[q]['ent'] = e1
                reltd[q]['cand'] = {}

            reltd[q]['cand'][e2] = label

    print('    Done loading relatedness {}. Num queries = {}'
          .format(set_type, len(reltd)))
    print('Writing dict File for future usage. Next time relatedness dataset '
          'will load faster!')
    torch.save(reltd, rel_dict)
    print('    Done saving.')
    return reltd


# Extracts all entities in the relatedness set, either candidates or :
def extract_reltd_ents(reltd):
    reltd_ents_direct = {}
    for v in reltd:
        reltd_ents_direct[reltd[v]['ent']] = 1
        for e2 in reltd[v]['cand']:
            reltd_ents_direct[e2] = 1
    return reltd_ents_direct


# computes rltd scores based on ground truth labels
def compute_ideal_rltd_scores(reltd):
    scores = {}
    for q in reltd:
        scores[q] = {}
        for e2 in reltd[q]['cand']:
            scores[q][e2] = reltd[q]['cand'][e2]
        scores[q] = sorted(scores[q].items(), key=lambda kv: kv[1],
                           reverse=True)
    return scores


# Mean Average Precision:
# https://en.wikipedia.org/wiki/Information_retrieval#Mean_average_precision
def compute_map(scores, reltd):
    sum_avgp = 0.0
    num_queries = 0
    for q in scores:
        sum_precision = 0.0
        num_rel_ents_so_far = 0
        num_ents_so_far = 0.0
        for kv in scores[q]:
            e2 = kv[0]
            label = reltd[q]['cand'][e2]
            num_ents_so_far = num_ents_so_far + 1.0
            if label == 1:
                num_rel_ents_so_far += 1
                precision = num_rel_ents_so_far / num_ents_so_far
                sum_precision += precision

        avgp = sum_precision / num_rel_ents_so_far
        sum_avgp += avgp
        num_queries += 1

    assert num_queries == len(reltd)
    return sum_avgp / num_queries


# computes rltd scores based on a given entity_sim function
def compute_e2v_rltd_scores(reltd, entity_sim):
    scores = {}
    for q in reltd:
        scores[q] = {}
        for e2 in reltd[q]['cand']:
            score = entity_sim(reltd[q]['ent'], e2)
            scores[q][e2] = score
        scores[q] = sorted(scores[q].items(), key=lambda kv: kv[1],
                           reverse=True)
    return scores


# NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
def compute_DCG(k, q, scores_q, reltd):
    dcg = 0.0
    i = 0
    for kv in scores_q:
        e2 = kv[0]
        label = reltd[q]['cand'][e2]
        i += 1
        if label == 1 and i <= k:
            dcg += (1.0 / log(max(2, i) + 0.0, 2))
    return dcg


def compute_NDCG(k, scores, reltd, ideals_rltd_scores):
    sum_ndcg = 0.0
    num_queries = 0
    for q in scores:
        dcg = compute_DCG(k, q, scores[q], reltd)
        idcg = compute_DCG(k, q, ideals_rltd_scores[q], reltd)
        assert dcg <= idcg, '{} {}'.format(dcg, idcg)
        sum_ndcg = sum_ndcg + (dcg / idcg)
        num_queries = num_queries + 1

    assert num_queries == len(reltd)
    return sum_ndcg / num_queries
