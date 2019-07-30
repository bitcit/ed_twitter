# python3 -m data_gen.conll_from_tweets

# CONLL format:

# -DOCSTART- (tweet_id
# word1
# word2
# mention1.1    B    full mention    full_entity    wikilink    000    000
# mention1.2    I    full mention    full_entity    wikilink    000    000
# word3
# mention2.1    B    full mention    full_entity    wikilink    000    000
#
# -DOCSTART- (tweet_id
# ...

import argparse
import os
import requests

from xml.etree import ElementTree

from utils.utils import split_in_words

parser = argparse.ArgumentParser()
parser.add_argument('-root_data_dir', default='data/',
                    help='Root path of the data, $DATA_PATH.')
parser.add_argument('-twitter_input',
                    default='basic_data/microposts2016/dev/')
parser.add_argument('-twitter_in_format', default='dir', help='xml|dir')
parser.add_argument('-out_file',
                    default='generated/microposts2016-dev.conll')
args = parser.parse_args()


def wikilink_from_dbpedia(dbpedia_link):
    wiki_pattern = 'wikipedia.org'
    l = len('http://dbpedia.org/resource/')
    dbpedia_ent_name = dbpedia_link[l:]
    json_path = 'http://dbpedia.org/data/{}.json'.format(dbpedia_ent_name)

    wikilink = ''
    try:
        data = requests.get(json_path).json()
    except:
        print('Could not retrieve json from {}'.format(json_path))
        return wikilink, dbpedia_ent_name

    if dbpedia_link in data:
        ent_data = data[dbpedia_link]

        topic_of_ont = 'http://xmlns.com/foaf/0.1/isPrimaryTopicOf'
        if topic_of_ont in ent_data:
            wikilink = ent_data[topic_of_ont][0]['value']
            if wiki_pattern not in wikilink:
                wikilink = ''

        derived_from_ont = 'http://www.w3.org/ns/prov#wasDerivedFrom'
        if not wikilink:
            if derived_from_ont in ent_data:
                wikilink = ent_data[derived_from_ont][0]['value']
                if wiki_pattern not in wikilink:
                    wikilink = ''

    if not wikilink:
        for k in data.keys():
            if wiki_pattern in k:
                wikilink = k
                break

    if not wikilink:
        print('Could not find wikilink for {}'.format(dbpedia_link))

    return wikilink, dbpedia_ent_name


def process_tweet(tweet_words, tweet_text, start_idx, end_idx, link,
                  corr_count):
    wikilink = ''

    if 'wikipedia.org' in link:
        wikilink = link
        full_entity = link[len('http://en.wikipedia.org/wiki/'):]
    elif "dbpedia.org" in link:
        wikilink, full_entity = wikilink_from_dbpedia(link)

    if not wikilink:
        return tweet_words, corr_count

    mnt_text = tweet_text[start_idx:end_idx]

    # Pointing to 1st word of the mention
    mnt_strt_idx = len(split_in_words(tweet_text[:start_idx]))
    # Pointing to the next word after mention (if there's any)
    mnt_end_idx = len(split_in_words(tweet_text[:end_idx]))

    # Bad index in dataset, find correct mention
    # Assumption: happens only when mention is 1 word
    if mnt_strt_idx == mnt_end_idx:
        while mnt_strt_idx < len(tweet_words):
            if tweet_words[mnt_strt_idx] == mnt_text:
                mnt_end_idx = mnt_strt_idx + 1
                break
            mnt_strt_idx += 1

    mnt_dtls = [mnt_text, full_entity, wikilink, '000', '000']
    pos = 'B'
    while mnt_strt_idx < mnt_end_idx:
        tweet_words[mnt_strt_idx] += '\t' + '\t'.join([pos] + mnt_dtls)
        pos = 'I'
        mnt_strt_idx += 1
        corr_count += 1

    return tweet_words, corr_count


def process_twitter_xml(xml_path, ouf):
    tree = ElementTree.parse(xml_path)
    tweets = tree.getroot().getchildren()[1]
    for tweet in tweets.getchildren():
        tweet_id = tweet.find('TweetId').text

        tweet_text = tweet.find('TweetText').text
        tweet_words = split_in_words(tweet_text)
        tweet_mentions = tweet.find('Mentions').getchildren()

        if len(tweet_mentions) < 1:
            continue

        correct_mnt_count = 0

        # Dict of mentions with word index as key
        # mentions = {}
        for mention in tweet_mentions:
            mnt_text = mention.find('Text').text
            start_idx = int(mention.find('StartIndx').text)
            end_idx = start_idx + len(mnt_text)
            link = mention.find('Entity').text

            tweet_words, correct_mnt_count = process_tweet(tweet_words,
                                                           tweet_text,
                                                           start_idx, end_idx,
                                                           link,
                                                           correct_mnt_count)

        if correct_mnt_count == 0:
            continue

        ouf.write('-DOCSTART- ({}\n'.format(tweet_id))
        ouf.write('\n'.join(tweet_words) + '\n' + '\n')


def load_mentions(mentions_path):
    mentions = {}

    with open(mentions_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            parts = line.split('\t')
            tweet_id = parts[0]
            start_idx = int(parts[1])
            end_idx = int(parts[2])
            link = parts[3]

            if tweet_id not in mentions:
                mentions[tweet_id] = {}

            mentions[tweet_id][start_idx] = (end_idx, link)

    return mentions


def process_twitter_microposts(input_dir, ouf):
    data_type = input_dir.split('/')[-2]
    tweets_path = '{}NEEL2016-{}.tsv'.format(input_dir, data_type)
    mentions_path = '{}NEEL2016-{}_neel.gs'.format(input_dir, data_type)

    all_mentions = load_mentions(mentions_path)

    with open(tweets_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            parts = line.split('\t')
            tweet_id = parts[0]
            tweet_text = parts[1]

            if tweet_id not in all_mentions:
                continue

            corr_mnt_count = 0

            tweet_mentions = all_mentions[tweet_id]
            mentions_order = list(tweet_mentions.keys())
            mentions_order.sort()

            tweet_words = split_in_words(tweet_text)

            for start_idx in mentions_order:
                end_idx, link = tweet_mentions[start_idx]
                tweet_words, corr_mnt_count = process_tweet(tweet_words,
                                                            tweet_text,
                                                            start_idx,
                                                            end_idx,
                                                            link,
                                                            corr_mnt_count)

            if corr_mnt_count == 0:
                continue

            ouf.write('-DOCSTART- ({}\n'.format(tweet_id))
            ouf.write('\n'.join(tweet_words) + '\n' + '\n')


ouf = open(os.path.join(args.root_data_dir, args.out_file), 'w')

if args.twitter_in_format == 'xml':
    twitter_xml_path = os.path.join(args.root_data_dir, args.twitter_input)
    process_twitter_xml(twitter_xml_path, ouf)
else:
    twitter_input_dir = os.path.join(args.root_data_dir, args.twitter_input)
    process_twitter_microposts(twitter_input_dir, ouf)

ouf.close()
