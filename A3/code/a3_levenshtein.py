import os
import numpy as np
import fnmatch
import string
import re
dataDir = '/u/cs401/A3/data/'
#dataDir = '../data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 celements (uint8).
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """

    len_r = len(r)
    len_h = len(h)
    if len_h == 0:
        return 1.0 ,0, 0, len_r
    if len_r == 0:
        return float("Inf") ,0, len_h,0

    #print(len_h, len_r)
    B =np.zeros((len_r+1 ,len_h+1))
    R =np.zeros((len_r+1 ,len_h+1))

    R[0,:] = float("inf")
    R[:,0] = float("inf")
    R[0,0] = 0
    #print("{} {}".format(len_r, len_h))

    for i in range(1,len_r+1):
        for j in range(1,len_h+1):
            dele = R[i-1,j] +1
            miss = R[i-1,j-1] +1
            hit = R[i-1,j-1]
            if(r[i-1] !=h[j-1]):
                hit = hit +1
            ins = R[i,j-1] +1
            R[i,j] = np.min([dele,ins,miss,hit])
            B[i,j] = np.argmin([dele,ins,miss,hit])
    n = len_r
    m = len_h

    dis_counter = np.zeros((4,1))

    while True:
        index = int(B[n,m])
        dis_counter[index] = dis_counter[index]+1
        n = n -(index!=2)
        m = m-(index!=1)
        #print("n:{}, m:{}, index:{}".format(n,m, index))
        if n ==0 or m ==0:
            break
    d = dis_counter[0]
    i = dis_counter[1]
    s = dis_counter[2]

    return float(s+i+d)/ float(len_r), s, i ,d






def process_lines(line) :

    words_parrten = re.compile(r"[<|\[].*[>\]]")
    pun_parrten = re.compile(r"[{}]".format(string.punctuation))
    line = re.sub(words_parrten, " ", line)
    line = re.sub(pun_parrten, " ", line)
    return line.lower()




if __name__ == "__main__":
    asr_output = open("asrDiscussion.txt","a")
    file_content = {}
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            #print(speaker)
            file_content[speaker] = {}
            speaker_path = os.path.join(dataDir, speaker)
            #print(speaker_path)
            files = fnmatch.filter(os.listdir(speaker_path), '*txt')
            if(len(files) <3): continue
            for i in files:
                for line in open(os.path.join(speaker_path,i), 'r').readlines():
                    source = "real"
                    if "oogle" in i:
                        source = "Google"
                    if "aldi" in i:
                        source = "Kaldi"


                    file_content[speaker][source] = process_lines(" ".join(line.split()[2:])) + "|"

            if(len(file_content[speaker].keys()) <3):continue

            if len(file_content[speaker]["Google"]) ==0 or len(file_content[speaker]["Kaldi"]) ==0:
                continue
            #print(file_content[speaker]["real"])
            #print(file_content[speaker]["Google"])
            #google
            real = file_content[speaker]["real"].split("|")
            google = file_content[speaker]["Google"].split("|")
            kaldi = file_content[speaker]["Kaldi"].split("|")
            for k in range(len(real)):
                wer,s,i,d = Levenshtein(real[k].split(),google[k].split())
                print("[{}] [Google] [{}] [{}] S:[{}], I:[{}], D:[{}]".format(
                    speaker,file_content[speaker]["real"], wer,s,i,d
                ))
            #print("test")
            #kalid
                wer, s, i, d = Levenshtein(real[k].split(),kaldi[k].split())
                print("[{}] [Kaldi] [{}] [{}] S:[{}], I:[{}], D:[{}]".format(
                    speaker, file_content[speaker]["real"], wer, s, i, d
                ))

