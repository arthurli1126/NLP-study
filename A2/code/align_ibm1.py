from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import os

def read_lan_file(train_dir, num_sen, lan):
    curr_num_sen = 0
    ret_words = {}
    for subdir, dirs, files in os.walk(train_dir):
        for file in files:
            if "."+lan in file:
                fullFile = os.path.join(subdir, file)
                print("Peprocessing " + fullFile)
                fp = open(fullFile)
                # preocess each sentence
                for line in fp:
                    outsen = preprocess(line.strip(), lan)
                    ret_words[curr_num_sen] = outsen.split()
                    curr_num_sen += 1
                    if curr_num_sen==num_sen:
                        return ret_words



def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
	Implements the training of IBM-1 word alignment algoirthm. 
	We assume that we are implemented P(foreign|english)
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model

	OUTPUT:
	AM :			(dictionary) alignment model structure
	
	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
	is the computed expectation that the foreign_word is produced by english_word.
	
			LM['house']['maison'] = 0.5
	"""
    AM = {}
    # Read training data
    print("data_dir : {}".format(train_dir))
    eng,fre = read_hansard(train_dir,num_sentences)
    # Initialize AM uniformly
    assert(len(eng)== len(fre))


    
    # Iterate between E and M steps

    

    return AM
    
# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
	Read up to num_sentences from train_dir.
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	
	
	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.
	
	Make sure to read the files in an aligned manner.
	"""
    eng = read_lan_file(train_dir, num_sentences, "e")
    fre = read_lan_file(train_dir, num_sentences, "f")
    return eng, fre

def initialize(eng, fre):
    """
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
    AM ={}
    for i in range(len(eng)):
        eng_sen = eng[i]
        fre_sen = fre[i]
        for j in range(len(eng_sen)):



    
def em_step(t, eng, fre):
    """
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""
	# TODO