from lm_train import *
from log_prob import *
from preprocess import *
from collections import OrderedDict,Counter
import pickle
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
    return ret_words



def align_ibm1(train_dir, num_sentences, max_iter, fn_AM="./temp_am.pickle"):
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
    AM = initialize(eng,fre)
    for i in range(max_iter):
        AM = em_step(AM,eng,fre)

    with open(fn_AM+'.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
        eng_sen = list(OrderedDict.fromkeys(eng[i]))
        fre_sen = list(OrderedDict.fromkeys(fre[i]))
        for j in range(1,len(eng_sen)-1):
            eng_word = eng_sen[j]
            for k in range(1,len(fre_sen)-1):
                fre_word= fre_sen[k]
                #print("processing eng:{}, fre:{}".format(eng_word,fre_word))
                if eng_word not in AM.keys():
                    #print(1)
                    AM[eng_word] ={fre_word:1}
                    continue
                if fre_word not in AM[eng_word].keys():
                    #print(2)
                    count = len(AM[eng_word])+1
                    AM[eng_word][fre_word] = 1/count
                    for h in AM[eng_word].keys():
                        AM[eng_word][h] = 1/count

    AM["SENTSTART"] = {"SENTSTART":1}
    AM["SENTEND"] = {"SENTEND":1}
    return AM

    
def em_step(t, eng, fre):
    """
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""
	# TODO
    tcount = {}
    total = {}
    #for each sentence pairs
    for i in range(len(eng)):
        eng_sen = eng[i]
        fre_sen = fre[i]
        eng_counts = Counter(eng_sen)
        fre_counts = Counter(fre_sen)
        uniq_eng_sen = list(OrderedDict.fromkeys(eng[i]))
        uniq_fre_sen = list(OrderedDict.fromkeys(fre[i]))
        #for each unique word f in F
        for j in range(1,len(uniq_fre_sen)-1):
            f = uniq_fre_sen[j]
            denom_c = 0
            #for each unique word e in E
            for k in range(1,len(uniq_eng_sen)-1):
                e = uniq_eng_sen[k]
                denom_c = denom_c + t[e][f]*fre_counts[f]
            #for each unique word e in E
            for k in range(1,len(uniq_eng_sen)-1):
                e = uniq_eng_sen[k]
                to_Add = t[e][f]*fre_counts[f]*eng_counts[e]/denom_c
                if e not in tcount.keys():
                    tcount[e] = {f:0}
                    total[e] = 0
                if f not in tcount[e].keys():
                    tcount[e][f] =0
                tcount[e][f] = tcount[e][f] + to_Add
                total[e] += to_Add
    #for each eng word in total
    for e in total.keys():
        for f in tcount[e].keys():
            t[e][f] = tcount[e][f]/total[e]
    return t