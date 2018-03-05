from preprocess import *
from lm_train import *
from math import log

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing
	
	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary
	
	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""
	
	#TODO: Implement by student.
    log_prob = 0
    words = sentence.split(" ")
    bi_log_prob = float("-inf")

    for i in range(len(words)-1):
        curr_word = words[i]
        next_word = words[i+1]
        count_curr = 0
        count_next = 0

        if curr_word in LM["uni"].keys():
            count_curr = LM["uni"][curr_word]
            if next_word in LM["bi"][curr_word].keys():
                count_next = LM["bi"][curr_word][next_word]

        if smoothing is True:
            bi_log_prob = log(count_next + delta/ (count_curr + (delta*vocabSize)), 2)
        elif count_next!=0 :
            bi_log_prob = log(count_next /count_curr, 2)

        log_prob = log_prob + bi_log_prob
            
    return log_prob