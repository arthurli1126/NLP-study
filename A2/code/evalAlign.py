from decode import *
import numpy as np
from math import log
from preprocess import *
from lm_train import *
from log_prob import *
from align_ibm1 import *
from BLEU_score import *
import pickle
import os

#dev env
# train_dir = "../Hansard/Training/"
# test_dir = "../Hansard/Testing/"
# fn_lme="./hansard_eng.pickle"
# fn_lmf="./hansard_fre.pickle"
# testF="../Hansard/Testing/Task5.f"
# teste="../Hansard/Testing/Task5.e"
# google_teste="../Hansard/Testing/Task5.google.e"

#prod env

train_dir = '/u/cs401/A2_SMT/data/Hansard/Training'
test_dir = '/u/cs401/A2_SMT/data/Hansard/Testing'
fn_lme="./hansard_eng.pickle"
fn_lmf="./hansard_fre.pickle"
testF= '/u/cs401/A2_SMT/data/Hansard/Testing/Task5.f'
teste= '/u/cs401/A2_SMT/data/Hansard/Testing/Task5.e'
google_teste='/u/cs401/A2_SMT/data/Hansard/Testing/Task5.google.e'


eval_log = "Task5"
max_iter =10

def read_file(file, lan):
    ret_sens = []
    print("Peprocessing " + file)
    fp = open(file)
    # preocess each sentence
    for line in fp:
        ret_sens.append(preprocess(line.strip(), lan))
    return ret_sens



def main():
    lme = pickle.load(open(fn_lme,"rb"))
    lmf = pickle.load(open(fn_lmf,"rb"))
    vocabSize = len(lme)
    data_size = [1000,10000,15000,30000]
    AMs = []
    for i in data_size:
        AMs.append(align_ibm1(train_dir, i, max_iter))

    f_sens=read_file(testF,"f")
    e_sens=read_file(teste,"e")
    google_e_sens=read_file(google_teste,"e")

    for i in range(len(f_sens)):
        print("processing:{}".format(f_sens[i]))
        fre = f_sens[i]
        ref_1 = e_sens[i]
        ref_2 = google_e_sens[i]
        print("ref1:{} ref2:{}".format(ref_1,ref_2))
        for j in range(len(data_size)):
            eng = decode(fre,lme,AMs[j])
            print("AM{} translation:{}".format(j,eng))
            for n in range(1,4):
                bleu = BLEU_score(eng, [ref_1,ref_2], n)
                print("{} bleu:{}".format(n,bleu))

if __name__ == '__main__':
    main()

