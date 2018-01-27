import numpy as np
import sys
import argparse
import os
import json
import string
import csv

'''
Load supported file
'''
#fp_path = '/u/cs401/Wordlists/First-person'
dev_fp_path = '../Wordlists/First-person'
fp_file = open(dev_fp_path)
fp = fp_file.readlines()
fp = [i.replace("\n","") for i in fp]
#sp_path = '/u/cs401/Wordlists/Second-person'
dev_sp_path = '../Wordlists/Second-person'
sp_file = open(dev_sp_path)
sp = sp_file.readlines()
sp = [i.replace("\n","") for i in sp]
#tp_path = '/u/cs401/Wordlists/Third-person'
dev_tp_path = '../Wordlists/Third-person'
tp_file = open(dev_tp_path)
tp = tp_file.readlines()
tp = [i.replace("\n","") for i in tp]
#slang_path = '/u/cs401/Wordlists/Slang'
dev_slang_path = '../Wordlists/Slang'
slang_file = open(dev_slang_path)
slang = slang_file.readlines()
slang = [i.replace("\n","") for i in slang]
#slang_path = '/u/cs401/Wordlists/Slang'
dev_slang_path = '../Wordlists/Slang'
slang_file = open(dev_slang_path)
slang = slang_file.readlines()
slang = [i.replace("\n","") for i in slang]
#BNG_path = '/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv'
dev_BNG_path = '../Wordlists/BristolNorms+GilhoolyLogie.csv'
BNG_file = csv.reader(open(dev_BNG_path))
BNG = np.array([row for row in BNG_file][1:])
bng_dict= dict(zip(BNG[:,1],BNG[:,3:6]))
#RW_path = '/u/cs401/Wordlists/Ratings_Warriner_et_al.csv'
dev_RW_path = '../Wordlists/Ratings_Warriner_et_al.csv'
RW_file = csv.reader(open(dev_RW_path))
RW = np.array([row for row in RW_file][1:])
rw_dict= dict(zip(RW[:,1],RW[:,[2,5,8]]))


#some helpful tags for future regex
common_nouns = ['NN', 'NNS']
proper_nouns = ['NNP', 'NNPS']
adverbs = ['RB', 'RBR', 'RBS']
wh = ['WDT', 'WP', 'WP$', 'WRB']



def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features
    '''

    feats = np.zeros(173)
    comment = comment.split(" ")
    len_comment = 0.0
    no_of_token = 0.0
    no_of_sen = 0.0
    no_of_pun = 0.0
    aoa = []
    img = []
    fam = []
    vmean = []
    amean = []
    dmean = []



    #TODO need to use regex seems like
    for i in comment:
        #some feature need to calculate regardless of words
        len_comment += len(i.split('/')[0])
        no_of_token += 1
        #Todo just figured need to spit word anyways might need to change it in the future
        word = i.split('/')[0]
        #place holder incase key not found
        pl = [0,0,0]
        aoa.append(float(bng_dict.get(word,pl)[0]))
        img.append(float(bng_dict.get(word,pl)[1]))
        fam.append(float(bng_dict.get(word, pl)[2]))
        vmean.append(float(rw_dict.get(word,pl)[0]))
        amean.append(float(rw_dict.get(word,pl)[1]))
        dmean.append(float(rw_dict.get(word,pl)[2]))

        if './.' in i:
            no_of_sen += 1
            no_of_pun +=1
            len_comment -= len(i.split('/')[0])
            #todo: AL need to rethink about it when have time
            continue

        #need to think a better way but should be enough for now
        if any(j +'/PRP' in i for j in fp) \
                or any(k +'/PRP$' in i for k in fp):
            feats[0] +=1
            continue
        if any(j + '/PRP' in i for j in sp) \
                or any(k + '/PRP$' in i for k in sp):
            feats[1] +=1
            continue
        if any(j + '/PRP' in i for j in tp) \
                or any(k + '/PRP$' in i for k in tp):
            feats[2] +=1
            continue
        if '/CC' in i:
            feats[3] +=1
            continue
        if '/VBD' in i:
            feats[4] +=1
            continue
        #TODO AL: need consider this kinda of stuff will/shall/going to/gonna/'ll
        if '/VBG' in i:
            feats[5] +=1
            continue
        if i == ',/,':
            feats[6] +=1
            len_comment -= len(i.split('/')[0])
            continue
        if i[0] in string.punctuation and i[1] in string.punctuation and len(i) >3:
            feats[7] +=1
            no_of_pun += 1
            len_comment -= len(i.split('/')[0])
            continue
        if any(n in i for n in common_nouns):
            feats[8] +=1
        if any(pn in i for pn in proper_nouns):
            feats[9] +=1
        if any(ad in i for ad in adverbs):
            feats[10] +=1
        if any(w in i for w in wh):
            feats[11] +=1
        if any(sl in i for sl in slang):
            feats[12] +=1
        if len(i.split('/')[0])>=3 and i.split('/')[0].isupper():
            feats[13] +=1

    #average length of sentence,tokens
    feats[14] = no_of_token/(no_of_sen if no_of_sen != 0 else 1)
    feats[15] = len_comment/\
                ((no_of_token-no_of_pun if no_of_token != 0 else 1)
                 if len_comment>0 else 1)
    feats[16] = no_of_sen
    feats[17] = np.mean(aoa)
    feats[18] = np.mean(img)
    feats[19] = np.mean(fam)
    feats[20] = np.std(aoa)
    feats[21] = np.std(img)
    feats[22] = np.std(fam)

    feats[23] = np.mean(vmean)
    feats[24] = np.mean(amean)
    feats[25] = np.mean(dmean)
    feats[26] = np.std(vmean)
    feats[27] = np.std(amean)
    feats[28] = np.std(dmean)


    #norm: average, sd






    return feats



def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))

    # TODO: your code here
    np.savez_compressed( args.output, feats)

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)

