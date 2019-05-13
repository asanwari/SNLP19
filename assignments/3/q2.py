import ngram_LM as lm
import sys
import time
from collections import Counter
import time

def print_frequent_ngrams(ngram_counter, top_n, ngram_length):

    sorted_ngrams = sorted(ngram_counter.items(), key=lambda count: count[1], reverse=True)
    print('The top {} {}-grams are:'.format(top_n, ngram_length))
    print(sorted_ngrams[0:top_n])


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
	print_frequent_ngrams(unigram_frequencies, 15, 1)

	bigram_frequencies = Counter(bigrams)
	bigrams = None
	bigram_frequencies = bigram_frequencies+unigram_frequencies
	print_frequent_ngrams(bigram_frequencies, 15, 2)


	# 2.1 c
	unigram_vocabulary = list(unigram_frequencies.keys())
	unigram_vocabulary = [i[0] for i in unigram_vocabulary]


	bigram_vocabulary = list(unigram_frequencies.keys())
	bigram_vocabulary = [i[0] for i in bigram_vocabulary]
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


if __name__ == "__main__":
	main()
