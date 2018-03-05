from preprocess import *
import pickle
import os

def lm_train(data_dir, language, fn_LM):
    """
	This function reads data from data_dir, computes unigram and bigram counts,
	and writes the result to fn_LM
	
	INPUTS:
	
    data_dir	: (string) The top-level directory continaing the data from which
					to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
	language	: (string) either 'e' (English) or 'f' (French)
	fn_LM		: (string) the location to save the language model once trained
    
    OUTPUT
	
	LM			: (dictionary) a specialized language model
	
	The file fn_LM must contain the data structured called "LM", which is a dictionary
	having two fields: 'uni' and 'bi', each of which holds sub-structures which 
	incorporate unigram or bigram counts
	
	e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
		  LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
    """
	
	# TODO: Implement Function

    language_model = {"uni":{}, "bi":{}}
    print("data_dir : {}".format(data_dir))
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            if "."+language in file:
                fullFile = os.path.join(subdir, file)
                print("Peprocessing " + fullFile)
                fp = open(fullFile)
                #preocess each sentence
                for line in fp:
                    outsen = preprocess(line.strip(),language).split(" ")
                    #print("outsen : {}".format(outsen))
                    #process unigram and bigram
                    for i in range(len(outsen)):
                        if outsen[i] not in language_model['uni'].keys():
                            language_model['uni'][outsen[i]] = 0
                            language_model['bi'][outsen[i]] = {}
                        language_model['uni'][outsen[i]] += 1
                        if i<(len(outsen)-1):
                            #print(language_model)
                            #print(i)
                            if outsen[i+1] not in language_model['bi'][outsen[i]].keys():
                                language_model['bi'][outsen[i]][outsen[i+1]] = 0
                            language_model['bi'][outsen[i]][outsen[i+1]] += 1

    #Save Model
    with open(fn_LM+'.pickle', 'wb') as handle:
        pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return language_model