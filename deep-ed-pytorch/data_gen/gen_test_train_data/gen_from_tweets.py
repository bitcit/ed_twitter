# python3 -m data_gen.gen_test_train_data.gen_from_tweets

# 3819 data/generated/test_train_data/Microposts2014_train.csv
# 1585 data/generated/test_train_data/Brian_Collection.csv
# 510 data/generated/test_train_data/Mena_Collection.csv

# grep -P 'GT:\t-1' path/file.csv | wc -l
# 672 Micro
# 579 Brian
# 87 Mena

# grep -P 'GT:\t-1' path/file.csv | grep "EMPTYCAND" | wc -l
# 437 Micro
# 313 Brian
# 43 Mena

import argparse
import os
import requests

from xml.etree import ElementTree

from entities.ent_name2id_freq.ent_name_id import EntityNameId
from data_gen.indexes.yago_crosswikis_wiki import YagoCrosswikisIndex

parser = argparse.ArgumentParser()
parser.add_argument('-root_data_dir', default='data/',
                    help='Root path of the data, $DATA_PATH.')
parser.add_argument('-wiki_redirects',
                    default='basic_data/wiki_redirects.txt',
                    help='Wikipedia redirects index.')
parser.add_argument('-wiki_name_id_map',
                    default='basic_data/wiki_name_id_map.txt',
                    help='Wikipedia name id map.')
parser.add_argument('-wiki_disambiguation',
                    default='basic_data/wiki_disambiguation_pages.txt',
                    help='Wikipedia disambiguation index.')
parser.add_argument('-merged_p_e_m',
                    default='generated/crosswikis_wikipedia_p_e_m.txt',
                    help='Merged entity-mention prior output.')
parser.add_argument('-yago_p_e_m',
                    default='generated/yago_p_e_m.txt',
                    help='YAGO entity-mention prior output.')
parser.add_argument('-twitter_input',
                    default='basic_data/microposts2016/dev/')
parser.add_argument('-twitter_in_format', default='dir', help='xml|dir')
parser.add_argument('-out_file',
                    default='generated/test_train_data/microposts2016-dev.csv')
args = parser.parse_args()

entity_name_id = EntityNameId(args)

crosswikis_path = os.path.join(args.root_data_dir, args.merged_p_e_m)
yago_p_e_m_path = os.path.join(args.root_data_dir, args.yago_p_e_m)
yago_crw_index = YagoCrosswikisIndex(crosswikis_path, yago_p_e_m_path)


def find_and_write_candidates(ouf, ctxt_str, mention, cur_ent_wikiid, ent_name):
    # Entity candidates from p(e|m) dictionary
    if mention in yago_crw_index.ent_p_e_m_index and \
            len(yago_crw_index.ent_p_e_m_index[mention]) > 0:
        sorted_cand = {}
        for ent_wikiid in yago_crw_index.ent_p_e_m_index[mention]:
            sorted_cand[ent_wikiid] = \
                yago_crw_index.ent_p_e_m_index[mention][ent_wikiid]

        sorted_cand = sorted(sorted_cand.items(), key=lambda kv: kv[1],
                             reverse=True)
        gt_pos = -1
        pos = 0
        candidates = []
        for ent_wikiid, p in sorted_cand:
            if pos > 100:
                break

            candidates.append('{},{:.3f},{}'
                              .format(ent_wikiid, p, entity_name_id
                                      .ent_from_wikiid(ent_wikiid)))

            pos += 1
            if ent_wikiid == cur_ent_wikiid:
                gt_pos = pos

        ctxt_str += '\t'.join(candidates) + '\tGT:\t'

        if gt_pos > 0:
            ouf.write('{}{},{}\n'.format(ctxt_str, gt_pos,
                                         candidates[gt_pos - 1]))
        elif cur_ent_wikiid != entity_name_id.unk_ent_wikiid:
            ouf.write('{}-1,{},{}\n'
                      .format(ctxt_str, cur_ent_wikiid, ent_name))
        else:
            ouf.write(ctxt_str + '-1\n')
    elif cur_ent_wikiid != entity_name_id.unk_ent_wikiid:
        ouf.write('{}EMPTYCAND\tGT:\t-1,{},{}\n'
                  .format(ctxt_str, cur_ent_wikiid, ent_name))
    else:
        ouf.write('{}EMPTYCAND\tGT:\t-1\n'.format(ctxt_str))


def wikiid_from_dbpedia(dbpedia_link):
    wikiid_ont = 'http://dbpedia.org/ontology/wikiPageID'
    dl = len('http://dbpedia.org/resource/')
    dbpedia_ent_name = dbpedia_link[dl:]
    json_path = 'http://dbpedia.org/data/{}.json'.format(dbpedia_ent_name)

    wikiid = 1
    try:
        data = requests.get(json_path).json()
    except:
        print('Could not retrieve json from {}'.format(json_path))
        return wikiid, dbpedia_ent_name

    if dbpedia_link in data:
        ent_data = data[dbpedia_link]
        if wikiid_ont in ent_data:
            wikiid = int(ent_data[wikiid_ont][0]['value'])

    if wikiid == 1:
        for k in data.keys():
            if 'http://dbpedia.org/resource/' in k:
                if wikiid_ont in data[k]:
                    ent_data = data[k]
                    wikiid = int(ent_data[wikiid_ont][0]['value'])
                    break

    if wikiid == 1:
        wikiid = entity_name_id.get_ent_wikiid_from_name(dbpedia_ent_name)

    if wikiid == 1:
        print('Could not find wikiid for {}'.format(dbpedia_link))

    return wikiid, dbpedia_ent_name


def wikiid_from_wikipedia(wiki_link):
    wiki_query = "http://en.wikipedia.org/w/api.php?action=query&prop=" \
                 "revisions&rvprop=content&rvsection=0&format=json&titles="
    wiki_pattern = "http://en.wikipedia.org/wiki/"
    wl = len(wiki_pattern)
    ent_name = wiki_link[wl:]
    wiki_query += ent_name

    wikiid = 1
    try:
        data = requests.get(wiki_query).json()
    except:
        print('Could not retrieve json from {}'.format(wiki_query))
        return wikiid, ent_name

    wikiid = int(list(data['query']['pages'].keys())[0])

    return wikiid, ent_name


def get_tweet_context(tweet_id, mention_text, tweet_text, start_idx, end_idx):
    ctxt_str = '\t'.join([tweet_id, tweet_id, mention_text]) + '\t'

    left_ctxt = tweet_text[:start_idx].split()
    if not left_ctxt:
        left_ctxt.append('EMPTYCTXT')
    ctxt_str += ' '.join(left_ctxt) + '\t'

    right_ctxt = tweet_text[end_idx:].split()
    if not right_ctxt:
        right_ctxt.append('EMPTYCTXT')
    ctxt_str += ' '.join(right_ctxt) + '\tCANDIDATES\t'
    return ctxt_str


def process_twitter_xml(xml_path, ouf):
    tree = ElementTree.parse(xml_path)
    tweets = tree.getroot().getchildren()[1]
    for tweet in tweets.getchildren():
        tweet_id = tweet.find('TweetId').text
        tweet_text = tweet.find('TweetText').text
        mentions = tweet.find('Mentions').getchildren()

        for mention in mentions:
            mention_text = mention.find('Text').text
            start_idx = int(mention.find('StartIndx').text)
            end_idx = start_idx + len(mention_text)

            ctxt_str = get_tweet_context(tweet_id, mention_text, tweet_text,
                                         start_idx, end_idx)

            link = mention.find('Entity').text
            if "wikipedia.org" in link:
                ent_wikiid, ent_name = wikiid_from_wikipedia(link)
            elif "dbpedia.org" in link:
                ent_wikiid, ent_name = wikiid_from_dbpedia(link)

            if not ent_wikiid:
                continue

            find_and_write_candidates(ouf, ctxt_str, mention_text, ent_wikiid,
                                      ent_name)
    return


def load_tweets(tweets_path):
    tweets = {}
    with open(tweets_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            parts = line.split('\t')
            tweet_id = parts[0]
            tweet_text = parts[1]
            tweets[tweet_id] = tweet_text

    return tweets


def process_twitter_microposts(input_dir, ouf):
    data_type = input_dir.split('/')[-2]
    tweets_path = '{}NEEL2016-{}.tsv'.format(input_dir, data_type)
    mentions_path = '{}NEEL2016-{}_neel.gs'.format(input_dir, data_type)

    tweets = load_tweets(tweets_path)

    with open(mentions_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            parts = line.split('\t')
            tweet_id = parts[0]
            start_idx = int(parts[1])
            end_idx = int(parts[2])
            link = parts[3]

            if tweet_id not in tweets:
                continue

            tweet_text = tweets[tweet_id]

            mention_text = tweet_text[start_idx:end_idx]

            ctxt_str = get_tweet_context(tweet_id, mention_text, tweet_text,
                                         start_idx, end_idx)

            if "wikipedia.org" in link:
                ent_wikiid, ent_name = wikiid_from_wikipedia(link)
            elif "dbpedia.org" in link:
                ent_wikiid, ent_name = wikiid_from_dbpedia(link)

            if not ent_wikiid:
                continue

            find_and_write_candidates(ouf, ctxt_str, mention_text, ent_wikiid,
                                      ent_name)


ouf = open(os.path.join(args.root_data_dir, args.out_file), 'w')

if args.twitter_in_format == 'xml':
    twitter_xml_path = os.path.join(args.root_data_dir, args.twitter_input)
    process_twitter_xml(twitter_xml_path, ouf)
else:
    twitter_input_dir = os.path.join(args.root_data_dir, args.twitter_input)
    process_twitter_microposts(twitter_input_dir, ouf)

ouf.close()
