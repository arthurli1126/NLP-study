import re
from functools import partial

punctuation = ".,:;()+-<>="
pun_pattern = re.compile(r"(\w*)((\w*)(\.|,|:|;|\(|\)|\+|-|<|>|=)(\s*))")
fre_con = re.compile(r"(l'|je t'|j'|s'|c'|qu')(\w+)")
spe_words = ["d\'abord", "d\'accord", "d\'ailleurs", "d\'habitude"]





def preprocess(in_sentence, language):
    """ 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation 
	
	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence
				   
	OUTPUT:
	out_sentence: (string) the modified sentence
    """
    out_sen = str
    in_sentence = in_sentence.lower()

    if language =="e":
        out_sen = pun_pattern.sub(lambda  m: m.group(1) + " " +m.group(2), in_sentence)

    if language=="f":
        out_sen = pun_pattern.sub(lambda m: m.group(1) + " " + m.group(2), in_sentence)
        out_sen = fre_con.sub(lambda  m: m.group(1) + " " + m.group(2),out_sen)

    return "SENTSTART " + out_sen +" SENTEND"



