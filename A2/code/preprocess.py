import re
from functools import partial

punctuation = ".,:;()+-<>="
pun_pattern = re.compile(r"(\w*)((\w*)(\.|,|:|;|\(|\)|\+|-|<|>|=)(\s*))(\s*)")

def sub_pun(match):
    sec_part = match.group(3)

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
    # TODO: Implement Function
    out_sen = str
    if language =="e":
        out_sen = pun_pattern.sub(lambda  m: m.group(1) + " " +
                                    m.group(2) +
                                    "" if m.group(3) == " " else " "
                                    + m.group(3) ,in_sentence)
    return out_sen



