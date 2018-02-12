import numpy as np
import argparse
import os
import json
import string
import csv
import time
import sys

'''
Load supported file
'''
fp_path = '/u/cs401/Wordlists/First-person'
fp_file = open(fp_path)
# dev_fp_path = '../Wordlists/First-person'
# fp_file = open(dev_fp_path)
fp = fp_file.readlines()
fp = [i.replace("\n", "") for i in fp]
sp_path = '/u/cs401/Wordlists/Second-person'
sp_file = open(sp_path)
# dev_sp_path = '../Wordlists/Second-person'
# sp_file = open(dev_sp_path)
sp = sp_file.readlines()
sp = [i.replace("\n", "") for i in sp]
tp_path = '/u/cs401/Wordlists/Third-person'
tp_file = open(tp_path)
# dev_tp_path = '../Wordlists/Third-person'
# tp_file = open(dev_tp_path)
tp = tp_file.readlines()
tp = [i.replace("\n", "") for i in tp]
slang_path = '/u/cs401/Wordlists/Slang'
slang_file = open(slang_path)
# dev_slang_path = '../Wordlists/Slang'
# slang_file = open(dev_slang_path)
slang = slang_file.readlines()
slang = [i.replace("\n", "") for i in slang if len(i) > 0]
BNG_path = '/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv'
BNG_file = csv.reader(open(BNG_path))
# dev_BNG_path = '../Wordlists/BristolNorms+GilhoolyLogie.csv'
# BNG_file = csv.reader(open(dev_BNG_path))
BNG = np.array([row for row in BNG_file][1:])
bng_dict = dict(zip(BNG[:, 1], BNG[:, 3:6]))
RW_path = '/u/cs401/Wordlists/Ratings_Warriner_et_al.csv'
RW_file = csv.reader(open(RW_path))
# dev_RW_path = '../Wordlists/Ratings_Warriner_et_al.csv'
# RW_file = csv.reader(open(dev_RW_path))
RW = np.array([row for row in RW_file][1:])
rw_dict = dict(zip(RW[:, 1], RW[:, [2, 5, 8]]))

# some helpful tags for future regex
common_nouns = ['NN', 'NNS']
proper_nouns = ['NNP', 'NNPS']
adverbs = ['RB', 'RBR', 'RBS']
wh = ['WDT', 'WP', 'WP$', 'WRB']


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features
    '''

    feats = np.zeros(174)
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

    # TODO need to use regex seems like
    for i in comment:
        if len(i) < 2:
            continue

        if './.' in i:
            no_of_sen += 1
            no_of_pun += 1
            no_of_token += 1
            # todo: AL need to rethink about it when have time
            continue

        # some feature need to calculate regardless of words
        len_comment += len(i.split('/')[0]) if len(i.split('/')[0]) > 0 else 1
        no_of_token += 1
        # Todo just figured need to spit word anyways might need to change it in the future
        word = i.split('/')[0] if i.split('/')[0] != '' else "/"
        # place holder in case key not found
        bng_norm = bng_dict.get(word, '')
        rw_norm = rw_dict.get(word, '')

        # print(word)
        # print(bng_norm)
        # print(rw_norm)
        if type(bng_norm) != str:
            aoa.append(float(bng_norm[0]))
            img.append(float(bng_norm[1]))
            fam.append(float(bng_norm[2]))
        if type(rw_norm) != str:
            vmean.append(float(rw_norm[0]))
            amean.append(float(rw_norm[1]))
            dmean.append(float(rw_norm[2]))

        # need to think a better way but should be enough for now
        # Todo need to change the code structure loop over token instead of steps

        if any(j == word for j in fp):
            feats[0] += 1
            continue
        if any(j == word for j in sp):
            feats[1] += 1
            continue
        if any(j == word for j in tp):
            feats[2] += 1
            continue
        if '/CC' in i:
            feats[3] += 1
            continue
        if '/VBD' in i:
            feats[4] += 1
            continue
        # TODO AL: need consider this kinda of stuff will/shall/going to/gonna/'ll
        if '/VBG' in i:
            feats[5] += 1
            continue
        if i == ',/,':
            feats[6] += 1
            len_comment -= len(i.split('/')[0])
            continue
        if i[0] in string.punctuation and i[1] in string.punctuation and len(i) > 3:
            feats[7] += 1
            no_of_pun += 1
            len_comment -= len(i.split('/')[0])
            continue
        if any(n in i for n in common_nouns):
            feats[8] += 1
        if any(pn in i for pn in proper_nouns):
            feats[9] += 1
        if any(ad in i for ad in adverbs):
            feats[10] += 1
        if any(w in i for w in wh):
            feats[11] += 1
        if any(sl == i for sl in slang):
            feats[12] += 1
        if len(i.split('/')[0]) >= 3 and i.split('/')[0].isupper():
            feats[13] += 1

    # average/std length of sentence,tokens, norms
    feats[14] = no_of_token / (no_of_sen if no_of_sen != 0 else 1)
    # in the situation that tokens are puns
    feats[15] = len_comment / (
    no_of_token - no_of_pun if no_of_token != no_of_pun and no_of_token != 0 else sys.maxsize)
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

    return feats


def liwc():
    feats_path = '/u/cs401/A1/feats'
    # dev_feats_path = '../feats'
    aid_index = []
    cid_index = []
    lid_index = []
    rid_index = []
    feats = []
    alt_feats = np.array([])
    center_feats = np.array([])
    left_feats = np.array([])
    right_feats = np.array([])
    for subdir, dir, files in os.walk(feats_path):
        for file in files:
            if 'txt' in file:
                full_file = open(os.path.join(subdir, file)).readlines()
                full_file = [i.replace("\n", "") for i in full_file]
                if 'Alt' in file:
                    aid_index = full_file
                if 'Center' in file:
                    cid_index = full_file
                if 'Left' in file:
                    lid_index = full_file
                if 'Right' in file:
                    rid_index = full_file
                if 'feats' in file:
                    feats = full_file
            if 'npy' in file:
                temp_feats = np.load(os.path.join(subdir, file))
                if 'Alt' in file:
                    alt_feats = temp_feats
                if 'Center' in file:
                    center_feats = temp_feats
                if 'Left' in file:
                    left_feats = temp_feats
                if 'Right' in file:
                    right_feats = temp_feats
        return aid_index, cid_index, lid_index, rid_index, \
               feats, alt_feats, center_feats, left_feats, right_feats


def find_liwc_feats(id, ids, features):
    id_index = ids.index(id)
    return features[id_index]


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173 + 1))
    cat_mapping = {
        'Left': 0,
        'Center': 1,
        'Right': 2,
        'Alt': 3
    }
    # TODO: your code here
    aid_index, cid_index, lid_index, rid_index, feats_id, alt_feats, center_feats, left_feats, right_feats = liwc()
    data_index = 0
    for i in data:
        # print("Processing comment: %s" %i['body'])
        print(data_index)
        feats_ex = extract1(i['body'])
        feats_ex[173] = cat_mapping.get(i['cat'], " ")
        if i['cat'] == 'Alt':
            feats_ex[29:173] = find_liwc_feats(i['id'], aid_index, alt_feats)
        if i['cat'] == 'Center':
            feats_ex[29:173] = find_liwc_feats(i['id'], cid_index, center_feats)
        if i['cat'] == 'Left':
            feats_ex[29:173] = find_liwc_feats(i['id'], lid_index, left_feats)
        if i['cat'] == 'Right':
            feats_ex[29:173] = find_liwc_feats(i['id'], rid_index, right_feats)
        feats[data_index] = feats_ex
        data_index += 1

    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
    t_0 = time.time()
    main(args)
    print(time.time() - t_0)

