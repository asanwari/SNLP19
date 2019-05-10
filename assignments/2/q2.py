# -*- coding: utf-8 -*-
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import sys
import os


alphabet_en = list('abcdefghijklmnopqrstuvwxyz')
alphabet_de = list(u'abcdefghijklmnopqrstuvwxyzäöüß')

# tokenizes text
def tokenize(untokenized_text):
    return untokenized_text.split()


# convert all alphabet characters to lower case
def convert_to_lower_case(text):
    return text.lower()


# keep only english and german alphabets, white spaces
def keep_alphabet(text):
    return re.sub(r'[^a-zA-ZäöüßÄÖÜ\s]', '', text)


def preprocess(text):
    return tokenize(keep_alphabet(convert_to_lower_case(text)))


# generates ngrams of length max_ngram_length .. 1
def char_ngrams(text, max_ngram_length):
    ngram_list = []
    # base case
    if max_ngram_length == 1:
        return list(text)
    # recursive case
    else:
        ngram_list.extend([text[i:i + max_ngram_length] for i in range(0, len(text) - max_ngram_length + 1)])
        ngram_list.extend(char_ngrams(text, max_ngram_length - 1))

    return ngram_list


def print_frequent_ngrams(ngram_counter, top_n, max_ngram_length):
    for i in range(1, max_ngram_length + 1):
        i_gram = {k: v for k, v in ngram_counter.items() if len(k) == i}
        sorted_igrams = sorted(i_gram.items(), key=lambda count: count[1], reverse=True)
        print('The top {} {}-grams are:'.format(top_n, i))
        print(sorted_igrams[0:top_n])

def prob_dist(n_gram_counts, alphabet, history):
    # pre calculate normalization, i.e. the denominator of the equation
    # set default distribution which would be max likelihood, i.e 1/len(alphabet) for all alphabets
    normalization = 0
    distribution = defaultdict()
    for s in alphabet:
        # normalization calculation
        word = history + s
        normalization += n_gram_counts[word]
        # distribution initialization
        distribution[s] = 1/len(alphabet)

    # distribution = defaultdict(lambda: 1/len(alphabet))
    if normalization == 0:
        return distribution
    else:
        for s in alphabet:
            word = history + s
            word_count = n_gram_counts[word]
            distribution[s] = word_count / normalization
    return distribution

def evaluate_prob_dist(probability_distribution):
    print('Asserting prob_dist sum = 1')
    prob_sum = sum(probability_distribution.values())
    # assertion that the distribution sums equal to 1+- error
    assert(1 + 1e-7 >= prob_sum >= 1 - 1e-7)
    print('Assertion successful!')


def plot_prob_dist(prob_dist, title = 'Probability distribution of all characters'):

    plot_data_x = []
    plot_data_y = []
    print('Plotting probability distribution...')
    for char_prob_pair in prob_dist.items():
        char = char_prob_pair[0]
        probability = char_prob_pair[1]
        plot_data_x.append(char)
        plot_data_y.append(probability)

    indices = [i for i in range(len(plot_data_x))]
    bar_plot = plt.bar(indices, plot_data_y)
    # plt.gca().tick_params(axis='x', which='major', pad=15)
    plt.xticks(indices, plot_data_x)
    plt.title(title)
    plt.show()


def entropy(prob_dist, alphabet):

    entrpy = 0
    for s in alphabet:
        prob = prob_dist[s]
        if prob == 0:
            continue
        prob_log_prob = -1 * prob * math.log2(prob_dist[s])
        entrpy += prob_log_prob

    return entrpy


def main():
    file_path = sys.argv[1]
    max_ngram_length = int(sys.argv[2])
    files = ['corpus.de','corpus.en']
    ngrams_count = defaultdict()
    for file in files:
        f = open(file_path + file, "r", encoding="utf8")
        # pre processing
        content = f.read()
        processed_content = preprocess(content)
        ngrams_count[file[7:]] = Counter()
        ngrams = []
        for i in processed_content:
            ngrams.extend(char_ngrams(i, max_ngram_length))
        ngrams_count[file[7:]] = Counter(ngrams)


    # 2.1 b
    print('\n\n\t\t------- Starting Q2.2 a ------- \n\n')
    print('\n\t\t------- Top 15 ngrams for deutsch ------- \n')
    print_frequent_ngrams(ngrams_count['de'], 15, max_ngram_length)
    print('\n\t\t------- Top 15 ngrams for english ------- \n')
    print_frequent_ngrams(ngrams_count['en'], 15, max_ngram_length)

    # 2.2 a
    # for history "da" in de
    print('\n\n\t\t------- Starting Q2.2 a ------- \n\n')
    evaluate_prob_dist(prob_dist(ngrams_count['de'],alphabet_de,'da'))
    # for history "tha" in en
    evaluate_prob_dist(prob_dist(ngrams_count['en'],alphabet_en,'tha'))
    # for gibberish
    evaluate_prob_dist(prob_dist(ngrams_count['en'],alphabet_en,'xzv'))


    #2.2 b
    print('\n\n\t\t------- Starting Q2.2 b ------- \n\n')
    history = ['','n','un','gun']
    probability_distributions = []
    
    title = 'Probability distribution of all characters, history = \"'
    for i in range(0, len(history)):
        print('------- history: {} ------- '.format(history[i]))
        probability_distributions.append(prob_dist(ngrams_count['de'],alphabet_de,history[i]))
        plot_prob_dist(probability_distributions[i],title + history[i]+'\"')


    # plot_prob_dist(probability_distribution)
    
    #2.2 c 
    print('\n\n\t\t------- Starting Q2.2 c ------- \n\n')
    for i in range(0, len(history)):
        print('\n------- history: {} ------- \n'.format(history[i]))
        ent = entropy(probability_distributions[i],alphabet_de)
        print('Entropy for history \"{}\": {}'.format(history[i],ent))

    # 2.2 e
    print('\n\n\t\t------- Starting Q2.2 e ------- \n\n')
    bigram_history = ['a','d','z','c']
    bigram_probability_distributions = []
    
    title = 'Probability distribution of all characters, history = \"'
    for i in range(0, len(bigram_history)):
        print('\n------- history: {} ------- \n'.format(bigram_history[i]))
        # plot
        bigram_probability_distributions.append(prob_dist(ngrams_count['de'],alphabet_de,bigram_history[i]))
        plot_prob_dist(bigram_probability_distributions[i],title + bigram_history[i]+'\"')
        # entropy
        ent = entropy(bigram_probability_distributions[i],alphabet_de)
        print('Entropy for history \"{}\": {}'.format(bigram_history[i],ent))


if __name__ == "__main__":
    main()
