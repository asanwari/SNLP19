import numpy as np
import pandas as pd
import csv
import re

print('preparing feature extraction module ...')
# preparation steps. This code runs only once
# 1. prepare lexicon datasets
    # read the files
nrc_unigrams = pd.read_csv('nrc_unigrams.tsv', sep='\t', names=['word','score','npos', 'nneg'], header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False)
nrc_bigrams = pd.read_csv('nrc_bigrams.tsv', sep='\t', names=['word','score','npos', 'nneg'], header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False)
sentiment140_unigrams = pd.read_csv('sentiment140_unigrams.tsv', sep='\t', names=['word','score','npos', 'nneg'], header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False)
sentiment140_bigrams = pd.read_csv('sentiment140_bigrams.tsv', sep='\t', names=['word','score','npos', 'nneg'], header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False)

    # drop extra columns and convert to dict, where key is uni/bigram and value is the score (whehter it is postive or negative). NOTE: takes some time (< 1 min)
nrc_unigrams = nrc_unigrams.drop(['npos','nneg'],axis=1).set_index('word').to_dict(orient='index')
nrc_bigrams = nrc_bigrams.drop(['npos','nneg'],axis=1).set_index('word').to_dict(orient='index')
sentiment140_unigrams = sentiment140_unigrams.drop(['npos','nneg'],axis=1)[~sentiment140_unigrams.duplicated(subset='word', keep='first')].set_index('word').to_dict(orient='index')
sentiment140_bigrams = sentiment140_bigrams.drop(['npos','nneg'],axis=1).set_index('word').to_dict(orient='index')

for k, v in nrc_unigrams.items():
    nrc_unigrams[k] = v['score']

for k, v in nrc_bigrams.items():
    nrc_bigrams[k] = v['score']

for k, v in sentiment140_unigrams.items():
    sentiment140_unigrams[k] = v['score']

for k, v in sentiment140_bigrams.items():
    sentiment140_bigrams[k] = v['score']



# functions to extract each group of features
# each takes a list of words and returns a feature vector

def get_lexicon_features(sent):
    # lexicon features are for unigrams and bigrams from two different reference datasets: NRC and Sentiment140

    # 1. get the score for each uni/bigram in the tweet
    nrc_unigram_scores = []
    nrc_bigram_scores = []
    sentiment140_unigram_scores = []
    sentiment140_bigram_scores = []

    for word in sent:
        unigram = word.lower()
        if unigram in nrc_unigrams:
            nrc_unigram_scores.append(nrc_unigrams[unigram])
        if unigram in sentiment140_unigrams:
            sentiment140_unigram_scores.append(sentiment140_unigrams[unigram])

    for i in range(len(sent)-1):
        bigram = sent[i].lower() + ' ' + sent[i+1].lower()
        if bigram in nrc_bigrams:
            nrc_bigram_scores.append(nrc_bigrams[bigram])
        if bigram in sentiment140_bigrams:
            sentiment140_bigram_scores.append(sentiment140_bigrams[bigram])

    # 2. compute the feature vectors based on the scores.
    nrc_unigram_features = []
    nrc_bigram_features = []
    sentiment140_unigram_features = []
    sentiment140_bigram_features = []

        # a. num of scores > 0
    nrc_unigram_features.append(len([s for s in nrc_unigram_scores if s > 0]))
    nrc_bigram_features.append(len([s for s in nrc_bigram_scores if s > 0]))
    sentiment140_unigram_features.append(len([s for s in sentiment140_unigram_scores if s > 0]))
    sentiment140_bigram_features.append(len([s for s in sentiment140_bigram_scores if s > 0]))

        # b. sum of scores
    nrc_unigram_features.append(sum(nrc_unigram_scores))
    nrc_bigram_features.append(sum(nrc_bigram_scores))
    sentiment140_unigram_features.append(sum(sentiment140_unigram_scores))
    sentiment140_bigram_features.append(sum(sentiment140_bigram_scores))

        # c. max score. if there are no scores at all, then 0. -10 is the lowest possible score.
    nrc_unigram_features.append(max([-10 * len(nrc_unigram_scores)] + nrc_unigram_scores))
    nrc_bigram_features.append(max([-10 * len(nrc_bigram_scores)] + nrc_bigram_scores))
    sentiment140_unigram_features.append(max([-10 * len(sentiment140_unigram_scores)] + sentiment140_unigram_scores))
    sentiment140_bigram_features.append(max([-10 * len(sentiment140_bigram_scores)] + sentiment140_bigram_scores))

        # d. last score that is > 0. if no score > 0, then 0.
    nrc_unigram_features.append(([0]+[s for s in nrc_unigram_scores if s > 0])[-1])
    nrc_bigram_features.append(([0]+[s for s in nrc_bigram_scores if s > 0])[-1])
    sentiment140_unigram_features.append(([0]+[s for s in sentiment140_unigram_scores if s > 0])[-1])
    sentiment140_bigram_features.append(([0]+[s for s in sentiment140_bigram_scores if s > 0])[-1])

    # 3. return all features stacked
    return nrc_unigram_features + nrc_bigram_features + sentiment140_unigram_features + sentiment140_bigram_features


def get_hashtag_features(sent):
    num_hashtags = len([s for s in sent if s[0] == '#'])
    return [num_hashtags]

def get_punctuation_features(sent):
    # num of continguous sequences of ? or ! in the tweet
    num_contiguous_marks = 0
    for w in sent:
        finds = re.findall(r'((\?|\!){2,})', w)
        num_contiguous_marks += len(finds)

    # whether the last word contains ? or !
    last_contains_mark = 1 if '?' in sent[-1] or '!' in sent[-1] else 0

    return [num_contiguous_marks, last_contains_mark]

def get_all_caps_features(sent):
    # num of words with all caps
    all_caps_counter = 0
    for word in sent:
        if word != '@USER' and word == word.upper():
            all_caps_counter += 1

    return [all_caps_counter]

def get_elongated_words_features(sent):
    # num of words that are elongated. e.g. soooo
    elongated_words_counter = 0
    for word in sent:
        if re.search(r'(\w)\1\1', word):
            elongated_words_counter += 1

    return [elongated_words_counter]


def extract_all_features(tweet):
    sent = tweet.split()
    features = []
    features.extend(get_lexicon_features(sent))
    features.extend(get_hashtag_features(sent))
    features.extend(get_punctuation_features(sent))
    features.extend(get_all_caps_features(sent))
    features.extend(get_elongated_words_features(sent))

    return features
