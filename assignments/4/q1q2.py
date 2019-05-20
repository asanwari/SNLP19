from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import ngram_LM as lm



def phonemes_distribution(corpora, alpha=1):
    phonemes = defaultdict(lambda: 0)
    phoneme_counter = 0

    f = open(corpora, 'r', encoding='utf-8')
    text = f.readline()
    f.close()

    for char in text:
        if char == '\n':    # ignore new line charachter
            continue
        phoneme_counter += 1
        phonemes[char] = phonemes[char] + 1

    # update the dict to make it smoothed probability
    phoneme_size = len(phonemes)
    for phoneme in phonemes:
        phonemes[phoneme] = (phonemes[phoneme] + alpha) / (alpha * phoneme_size + phoneme_counter)

    phonemes.default_factory = lambda: alpha / (alpha * phoneme_size + phoneme_counter)

    return phonemes


def unified_phoneme_set(phoneme_list):
    phoneme_set = set()
    for phoneme in phoneme_list:
        phonemes = phoneme.keys()
        phoneme_set.update(phonemes)

    return list(phoneme_set)

def plot_distribution(distributions, names):
    num_rows = np.sqrt(len(distributions))
    for distribution, name, i in zip(distributions, names, list(np.arange(len(distributions)))):
        x = [x[0] for x in sorted(distribution.items(), key=lambda kv: kv[1])]
        y = [x[1] for x in sorted(distribution.items(), key=lambda kv: kv[1])]

        plt.subplot(num_rows,num_rows,i+1)
        plot = plt.bar(np.arange(len(x)), y)
        plt.xticks(np.arange(len(x)), x)
        plt.title('Phoneme distribution of ' + name)

    plt.show()

def kl_div(p, q, sample_space):
    kl = 0
    for phoneme in sample_space:
        prob_p = p[phoneme]
        prob_q = q[phoneme]

        log_p_q = math.log2(prob_p/prob_q)
        kl += prob_p * log_p_q
    return kl

def percentage_unseen(corpus, lm):
    unseen_frequency = 0
    for gram in corpus:
        gram_frequency = lm.ngram_counts[gram]
        if gram_frequency == 0:
            unseen_frequency += 1
    unseen_percentage = unseen_frequency / len(corpus)
    return unseen_percentage

def plot_perplexity_chart(corpus, lm, name):
    alphas = np.arange(0.1, 1.1, 0.1)
    perplexities = []
    for alpha in alphas:
        perplexity = lm.perplexity(corpus, alpha=alpha)
        perplexities.append(perplexity)

    fig, ax = plt.subplots()
    ax.plot(alphas, perplexities)
    ax.set(title='Perplexity vs. alpha for ' + name)
    plt.show()

def main():
    IPA_EN = 'ipa_corpus/corpus.ipa.en'
    IPA_ES = 'ipa_corpus/corpus.ipa.es'
    IPA_FR = 'ipa_corpus/corpus.ipa.fr'
    IPA_IT = 'ipa_corpus/corpus.ipa.it'
    TRAIN_CORPUS = 'lm_eval/corpus.sent.en.train'
    TEST_CORPUS_SIMPLE = 'lm_eval/simple.test'
    TEST_CORPUS_WIKI = 'lm_eval/wiki.test'
    YODA_CORPUS = 'lm_eval/yodish.sent'
    ENGLISH_CORPUS = 'lm_eval/english.sent'

    phonemes_en = phonemes_distribution(IPA_EN)
    phonemes_es = phonemes_distribution(IPA_ES)
    phonemes_fr = phonemes_distribution(IPA_FR)
    phonemes_it = phonemes_distribution(IPA_IT)
    phonemes_list = [phonemes_en, phonemes_es, phonemes_fr, phonemes_it]

    # 1.1 testing the distribution
    print('\n\n***************** 1.1 a **********************')
    print('Testing phoneme distributions:')
    print('Probability of phoneme "a" in English: {}, Spanish: {}, French: {}, Italian: {}'.format(phonemes_en['a'], phonemes_es['a'], phonemes_fr['a'], phonemes_it['a']))
    print('Probability of non-existing phoneme "1" in English: ', phonemes_en['1'])
    phonemes_en.pop('1')
    print('\nUnified phoneme set:')
    unified_set = unified_phoneme_set(phonemes_list)
    print(unified_set)

    # 1.1 plotting the four distributions
    plot_distribution(phonemes_list, ['English', 'Spanish', 'French', 'Italian'])

    # 1.1 computing kl-divergence
    print('\n\n***************** 1.1 d **********************')
    print('KL-Divergence matrix:')
    kl_div_en = [kl_div(phonemes_en, phonemes_en, unified_set), kl_div(phonemes_en, phonemes_es, unified_set), kl_div(phonemes_en, phonemes_fr, unified_set), kl_div(phonemes_en, phonemes_it, unified_set)]
    kl_div_es = [kl_div(phonemes_es, phonemes_en, unified_set), kl_div(phonemes_es, phonemes_es, unified_set), kl_div(phonemes_es, phonemes_fr, unified_set), kl_div(phonemes_es, phonemes_it, unified_set)]
    kl_div_fr = [kl_div(phonemes_fr, phonemes_en, unified_set), kl_div(phonemes_fr, phonemes_es, unified_set), kl_div(phonemes_fr, phonemes_fr, unified_set), kl_div(phonemes_fr, phonemes_it, unified_set)]
    kl_div_it = [kl_div(phonemes_it, phonemes_en, unified_set), kl_div(phonemes_it, phonemes_es, unified_set), kl_div(phonemes_it, phonemes_fr, unified_set), kl_div(phonemes_it, phonemes_it, unified_set)]
    kl_divs = pd.DataFrame([kl_div_en, kl_div_es, kl_div_fr, kl_div_it], columns=['English', 'Spanish', 'French', 'Italian'], index=['English', 'Spanish', 'French', 'Italian'])
    print(kl_divs)

    # 2.1 constructing lm
    print('\n\n***************** 2.1  **********************')
    train_unigrams = []
    train_bigrams = []
    f = open(TRAIN_CORPUS, 'r', encoding='utf-8')
    for sentence in f:
        train_unigrams.extend(lm.word_ngrams(sentence, 1))
        train_bigrams.extend(lm.word_ngrams(sentence, 2))
    train_unigram_frequencies = Counter(train_unigrams)
    train_bigram_frequencies = Counter(train_bigrams)
    train_unigram_vocabulary = list(train_unigram_frequencies.keys())
    train_unigram_vocabulary = [i[0] for i in train_unigram_vocabulary]

    train_bigram_vocabulary = [i for i in train_unigram_vocabulary]
    train_bigram_vocabulary.extend(['<s>','</s>'])

    unigram_lm = lm.ngram_LM(1, train_unigram_frequencies, train_unigram_vocabulary)
    bigram_lm = lm.ngram_LM(2, train_bigram_frequencies, train_bigram_vocabulary)

    test_simple_unigrams = []
    test_simple_bigrams = []
    f = open(TEST_CORPUS_SIMPLE, 'r', encoding='utf-8')
    for sentence in f:
        test_simple_unigrams.extend(lm.word_ngrams(sentence, 1))
        test_simple_bigrams.extend(lm.word_ngrams(sentence, 2))

    test_wiki_unigrams = []
    test_wiki_bigrams = []
    f = open(TEST_CORPUS_WIKI, 'r', encoding='utf-8')
    for sentence in f:
        test_wiki_unigrams.extend(lm.word_ngrams(sentence, 1))
        test_wiki_bigrams.extend(lm.word_ngrams(sentence, 2))

    # 2.1 c computing perplexity
    print('\n\n***************** 2.1 c  **********************')
    print('Perplexity Simple - Smoothed - Unigram: {}\tBigram: {}'.format(unigram_lm.perplexity(test_simple_unigrams, 0.2), bigram_lm.perplexity(test_simple_bigrams, 0.2)))
    print('Perplexity Simple - Unsmoothed - Unigram: {}\tBigram: {}'.format(unigram_lm.perplexity(test_simple_unigrams, 0), bigram_lm.perplexity(test_simple_bigrams, 0)))
    print('Perplexity Wiki - Smoothed - Unigram: {}\tBigram: {}'.format(unigram_lm.perplexity(test_wiki_unigrams, 0.2), bigram_lm.perplexity(test_wiki_bigrams, 0.2)))
    print('Perplexity Wiki - Unsmoothed - Unigram: {}\tBigram: {}'.format(unigram_lm.perplexity(test_wiki_unigrams, 0), bigram_lm.perplexity(test_wiki_bigrams, 0)))

    # 2.1 d computing unseen percentage
    print('\n\n***************** 2.1 d **********************')
    print('Unseen unigrams percentage of Simple: ', percentage_unseen(test_simple_unigrams, unigram_lm))
    print('Unseen bigrams percentage of Simple: ', percentage_unseen(test_simple_bigrams, bigram_lm))
    print('Unseen unigrams percentage of Wiki: ', percentage_unseen(test_wiki_unigrams, unigram_lm))
    print('Unseen bigrams percentage of Wiki: ', percentage_unseen(test_wiki_bigrams, bigram_lm))

    # 2.1 e

    plot_perplexity_chart(test_simple_unigrams, unigram_lm, 'Simple Unigram')
    plot_perplexity_chart(test_simple_bigrams, bigram_lm, 'Simple Bigram')
    plot_perplexity_chart(test_wiki_unigrams, unigram_lm, 'Wiki Unigram')
    plot_perplexity_chart(test_wiki_bigrams, bigram_lm, 'Wiki Bigram')

    # 2.2 a calculating sentence scores for yoda and english
    print('\n\n***************** 2.2 a  **********************')
    yoda_sentences = []
    english_sentences = []
    yoda_bigrams = []
    english_bigrams = []
    f = open(ENGLISH_CORPUS, 'r', encoding='utf-8')
    for line in f:
        if line == '\n':
            continue
        sentence = lm.word_ngrams(line, 2)
        english_sentences.append(line)
        english_bigrams.append(sentence)
    f = open(YODA_CORPUS, 'r', encoding='utf-8')
    for line in f:
        if line == '\n':
            continue
        sentence = lm.word_ngrams(line, 2)
        yoda_sentences.append(line)
        yoda_bigrams.append(sentence)
    eng_scores = []
    yoda_scores = []
    for eng_sentence, yoda_sentence in zip(english_bigrams, yoda_bigrams):
        eng_scores.append(round(bigram_lm.score_sentence(eng_sentence), 2))
        yoda_scores.append(round(bigram_lm.score_sentence(yoda_sentence), 2))
    eng_scores = np.array(eng_scores)
    yoda_scores = np.array(yoda_scores)

    score_matrix = pd.DataFrame({'Eng': english_sentences, 'Yoda': yoda_sentences, 'Eng Score': eng_scores, 'Yoda Score': yoda_scores, 'Diff': yoda_scores - eng_scores})
    print(score_matrix)


if __name__ == "__main__":
    main()
