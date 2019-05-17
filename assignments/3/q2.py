import ngram_LM as lm
import sys
import time
from collections import Counter, defaultdict
import time

def print_frequent_ngrams(ngram_counter, top_n, ngram_length):
    sorted_ngrams = sorted(ngram_counter.items(), key=lambda count: count[1], reverse=True)
    print('The top {} {}-grams are:'.format(top_n, ngram_length))
    print(sorted_ngrams[0:top_n])

def print_top_n(dictionary, top_n, message=""):
    sorted_dict = sorted(dictionary.items(), key=lambda count: count[1], reverse=True)
    print('\n--------------------\n')
    print(message,'\n')
    print(sorted_dict[0:top_n])
    print('\n--------------------\n\n')

def calculate_ttr(ngram_counter):
    types = len(ngram_counter.keys())
    tokens = sum(ngram_counter.values())
    return (types/tokens) * 100


def main():

    file_name = sys.argv[1]
    f = open(file_name, "r", encoding="utf8")

    # ex 2.1 b
    unigrams = []
    bigrams = []
    sentence_counter = 0
    for sentence in f:
        unigrams.extend(lm.word_ngrams(sentence, 1))
        bigrams.extend(lm.word_ngrams(sentence, 2))
        sentence_counter = sentence_counter +1

    unigram_frequencies = Counter(unigrams)
    unigrams = None
    print_top_n(unigram_frequencies, 15, 'The top 15 unigrams are: ')

    bigram_frequencies = Counter(bigrams)
    bigrams = None
    print_top_n(bigram_frequencies, 15, 'The top 15 bigrams are: ')


    # 2.1 c
    unigram_vocabulary = list(unigram_frequencies.keys())
    unigram_vocabulary = [i[0] for i in unigram_vocabulary]

    bigram_vocabulary = [i for i in unigram_vocabulary]
    bigram_vocabulary.extend(['<s>','</s>'])

    print('The ttr is {}'.format(calculate_ttr(unigram_frequencies)))

    # 2.1 d
    t1 = time.time()
    unigram_LM = lm.ngram_LM(1, unigram_frequencies, unigram_vocabulary)
    t2 = time.time()
    print('unigram LM made in {} secs'.format(t2-t1))
    print(unigram_LM.estimate_prob('', 'and'))
    unigram_LM.test_LM()
    t3 = time.time()
    print('unigram LM tested in {} secs'.format(t3-t2))

    bigram_LM = lm.ngram_LM(2, bigram_frequencies, bigram_vocabulary)
    t4 = time.time()
    print('bigram LM made in {} secs'.format(t4 - t3))
    print(bigram_LM.estimate_prob('of', 'the'))
    bigram_LM.test_LM()
    print('bigram LM tested in {} secs'.format(time.time() - t4))


    # 2.1 e
    histories = ['blue', 'green', 'white', 'black', 'natural', 'artificial', 'global', 'domestic']
    for history in histories:
        history_prob_dist = defaultdict()
        for word in bigram_LM.vocab:
            history_prob_dist[word] = bigram_LM.estimate_prob(history, word)
        print_top_n(history_prob_dist, 10, 'The top 10 for history {} are:'.format(history))


    # 2.2 b
    unigram_LM.test_smoohted_LM()
    bigram_LM.test_smoohted_LM()

    # 2.2 c
    translation1 = 'Yesterday was I at home'
    translation2 = 'Yesterday I was at home'
    translation3 = 'I was at home yesterday'

    ngrams1 = lm.word_ngrams(translation1, 2)
    ngrams2 = lm.word_ngrams(translation2, 2)
    ngrams3 = lm.word_ngrams(translation3, 2)

    score1 = bigram_LM.score_sentence(ngrams1)
    score2 = bigram_LM.score_sentence(ngrams2)
    score3 = bigram_LM.score_sentence(ngrams3)

    print('Score of "Yesterday was I at home": ', score1)
    print('Score of "Yesterday I was at home":', score2)
    print('Score of "I was at home yesterday":', score3)




if __name__ == "__main__":
    main()
