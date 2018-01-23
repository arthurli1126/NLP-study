import numpy as np
import sys
import argparse
import os
import json

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
#Tp_path = '/u/cs401/Wordlists/Third-person'
dev_tp_path = '../Wordlists/Third-person'
tp_file = open(dev_tp_path)
tp = tp_file.readlines()
tp = [i.replace("\n","") for i in tp]



def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features
    '''
    feats = np.zeros(173)
    comment = comment.split(" ")
    for i in comment:
        if i in fp:
            feats[1] +=1
            continue
        if i in sp:
            feats[2] +=1
            continue
        if i in tp:
            feats[3] +=1


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

