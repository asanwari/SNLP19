# coding: utf-8
# SNLP - SoSe 2019 - ASSINGMENT III 

import math
import re
from collections import defaultdict, Counter
from nltk import ngrams



def tokenize(text):
	"List all the word tokens (consecutive letters) in a text. Normalize to lowercase."
	return re.findall('[a-z]+', text.lower())

def word_ngrams(sent, n):
	"""Givne a sent as str return n-grams as a list of tuple"""
	
	# EXAMPLES 
	# > word_ngrams(tokenize('hello world'), 1)
	# [('hello',), ('world',)]
	# > word_ngrams(tokenize('hello world'), 2)
	# [('<s>', 'hello'), ('hello', 'world'), ('world', '</s>')]

	# YOUR CODE HERE
	tokenized_sent = tokenize(sent)
	if n != 1:
		tokenized_sent.insert(0, '<s>')
		tokenized_sent.append('</s>')
	return [tuple(tokenized_sent[i:i + n]) for i in range(0, len(tokenized_sent)-n+1)]


class ngram_LM:
	"""A class to represent a language model."""

	def __init__(self, n, ngram_counts, vocab, unk=False):
		""""Make a n-gram language model, given a vocab and
			data structure for n-gram counts."""
		
		self.n = n 
		
		self.vocab = vocab
		
		self.V = len(vocab)

		self.ngram_counts = ngram_counts

		if n == 1:
			self.total_counts = sum(self.ngram_counts.values())
		elif n == 2:
			self.history_count = defaultdict()
			for history in vocab:
				a = 'd'
				# sums over counts of all pairs (history, v) where v belongs to the vocabulary list
				# for all elements in vocabulary list
				# self.history_count[history] = sum([self.ngram_counts[tuple([history, v])] for v in self.vocab])


	# YOUR CODE HERE
	# START BY MAKING THE RIGHT COUNTS FOR THIS PARTICULAR self.n 

		
	   
   

	def estimate_prob(self, history, word):
		"""Estimate probability of a word given a history."""
		# YOUR CODE HERE

		if history == '':
			# unigram
			word_frequency = self.ngram_counts[tuple([word])]
			return word_frequency/self.total_counts

		else:
			# bigram
			word_frequency = self.ngram_counts[tuple([history, word])]
			# history_count = sum([self.ngram_counts[key] for key in self.ngram_counts if key[0] == history])
			# history_count = self.history_count[history]
			history_count = self.ngram_counts[tuple([history])]
			# print('his: {}',format(history))
			# print('his count {}'.format(history_count))
			return word_frequency/history_count

	
	def estimate_smoothed_prob(self, history, word, alpha = 0.5):
		"""Estimate probability of a word given a history with Lidstone smoothing."""

		if history == '':
			# unigram
			word_frequency = self.ngram_counts[tuple([word])]
			return (word_frequency + alpha)/(alpha*self.V +self.total_counts)

		else:
			# bigram
			word_frequency = self.ngram_counts[tuple([history, word])]
			# history_count = sum([self.ngram_counts[key] for key in self.ngram_counts if key[0] == history])
			# history_count = self.history_count[history]
			history_count = self.ngram_counts[tuple([history])]
			# print('his: {}',format(history))
			# print('his count {}'.format(history_count))
			return (word_frequency + alpha)/(alpha*self.V + history_count)

		
	   
	# YOUR CODE HERE
			

	def logP(self, history, word):
		"""Return base-2 log probablity."""

	# YOUR CODE HERE
				 


	def score_sentence(self, sentence):
		"""Given a sentence, return score."""
		
		# YOUR CODE HERE


 
	def test_LM(self):
		"""Test whether or not the probability mass sums up to one."""
		
		precision = 10**-8
				 
		if self.n == 1:
				 
			P_sum = sum(self.estimate_prob('', w) for w in self.vocab)
			
			assert abs(1.0 - P_sum) < precision, 'Probability mass does not sum up to one.'
				 
		elif self.n == 2:
			histories = ['the', 'in', 'at', 'blue', 'white']
				 
			for h in histories:
				 
				P_sum = sum(self.estimate_prob(h, w) for w in self.vocab)
				
				assert abs(1.0 - P_sum) < precision, 'Probability mass does not sum up to one for history' + h
					 
		print('TEST SUCCESSFUL!')



	def test_smoohted_LM(self):
		"""Test whether or not the smoothed probability mass sums up to one."""
		precision = 10**-8
				 
		if self.n == 1:
				 
			P_sum = sum(self.estimate_smoothed_prob('', w) for w in self.vocab)
			
			assert abs(1.0 - P_sum) < precision, 'Probability mass does not sum up to one.'
				 
		elif self.n == 2:
			histories = ['the', 'in', 'at', 'blue', 'white']
				 
			for h in histories:
				 
				P_sum = sum(self.estimate_smoothed_prob(h, w) for w in self.vocab)
				
				assert abs(1.0 - P_sum) < precision, 'Probability mass does not sum up to one for history' + h
					 
		print('TEST SUCCESSFUL!')
		# YOUR CODE HERE


# # ADD YOUR CODE TO COLLECT COUTNS AND CONSTRUCT VOCAB


# # ONCE YOU HAVE N-GRAN COUNTS AND VOCAB, 
# # YOU CAN BUILD LM OBJECTS AS ... 
# unigram_LM = ngram_LM(1, unigram_frequencies, vocabulary)


# # THEN TEST YOUR IMPLEMENTATION AS ..
# unigram_LM.test_LM() 
	


