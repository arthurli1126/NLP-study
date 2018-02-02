import sys
import argparse
import os
import json
import random
import html
import re
import string
import spacy

#todo: need to implement multithreading

#indir = '/u/cs401/A1/data/';
#sw_path = 'u/cs401/Wordlists/StopWords';
#ab_path = 'u/cs401/Wordlists/abbrev.english';
dev_dir = '../data/';
dev_sw_path ='../Wordlists/StopWords';
dev_ab_path ='../Wordlists/abbrev.english';

# load abbrevs file
abbrev_file = open(dev_ab_path)
abbrevs = abbrev_file.readlines()
abbrev_file.close()
abbrevs = [i.replace("\n", "") for i in abbrevs]
# load st_words file
st_words_file = open(dev_sw_path)
st_words = st_words_file.readlines()
st_words_file.close()
st_words = [i.replace("\n", "") for i in st_words]

#create nlp
nlp = spacy.load('en', disable=['parse', 'ner'])
punctuation = re.sub(r"[']",'',string.punctuation)

def preproc1(comment, steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''

    modComm = ''
    modComm_6 = ''
    modComm_8 = ''


    if 1 in steps:
        comment = remove_newline(comment)
    if 2 in steps:
        comment = replace_html_code(comment)
    if 3 in steps:
        comment = remove_urls(comment)
    if 4 in steps:
        comment = split_punctuation(comment,abbrevs)
    if 5 in steps:
        pass
    if 6 in steps:
        modComm_6, modComm_8 = spacy_support(comment)
        comment = modComm_6
    if 7 in steps:
        comment = remove_sd(comment,st_words)
    if 8 in steps:
        if 6 in steps:
            comment = modComm_8
        else:
            modComm_6, modComm_8 = spacy_support(comment)
            comment = modComm_8
    if 9 in steps:
        comment = add_newline_eos(comment)
    if 10 in steps:
        comment = convert_lc(comment)


    modComm = comment
    return modComm

'''
remove new line 
'''
def remove_newline(comment):
    return re.sub("\n"," ",comment)

'''
replace hmtl code with ascii
'''
def replace_html_code(comment):
    return html.unescape(comment)

'''
remove urls
'''
def remove_urls(comment):
    pattern = re.compile(r'(http[^\s]*|www[^\s]*)')
    comment = re.sub(pattern,'', comment)
    return comment

'''
split_punctuation
'''
def split_punctuation(comment,abbrevs):
    #change the period of abbrev to magic=xeq this is bad
    comment = re.sub(r"(\b)(" + "|".join(abbrevs) + r")",lambda m: m.group(1) + m.group(2).replace(".","xeqxeq"), comment)
    comment = re.sub(r"(\w)([{}])".format(punctuation), '\g<1> \g<2> ', comment).replace("xeqxeq", ".")
    comment = re.sub(r"\s+",' ',comment)
    return comment

def split_clitics(comment):
 #clitics list will be sub
    c_list = ['\'s',
              '\'re',
              '\'ve',
              '\'d',
              '\'t',
              '\'ll',
              '\'m',
              's\'']

    comment = re.sub('\'',' ',comment)
    return comment

'''
step6&8 spacy support 
'''
def spacy_support(comment):
    modcom_6 = ''
    modcom_8 = ''
    utt = nlp(comment)
    #to-do AL if token start with - dont lemmanize it
    for token in utt:
        modcom_6 = modcom_6 + " " + token.text + "/" + token.tag_
        #modcom_8 = modcom_8 + " " + token.lemma_ + "/" + token.tag_
        modcom_8 = modcom_8 + " " + (token.lemma_ if token.lemma_[0]!='-' else token.text) + "/" + token.tag_
    # AL-first character was space
    return modcom_6[1:],modcom_8[1:]


def remove_sd(comment, st_words):
    pattern = re.compile(r"\b("+r"|".join(st_words)+r")(\/\S*)?\b")
    comment = re.sub(pattern,'', comment+' ')
    comment = re.sub(r"\s+", ' ', comment)
    return comment

def add_newline_eos(comment):
    return re.sub(r"(\w)([.])",'\g<1>\g<2>\n', comment)

#this one is little tricky
def convert_lc(comment):
    return re.sub(r"(\b\S+)(/\S+\b)",lambda m: m.group(1).lower() + m.group(2), comment)



def main(args):

    allOutput = []

    for subdir, dirs, files in os.walk(dev_dir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            start_index = args.ID[0]%len(data)
            print("start index : %s" %start_index)
            data = data[start_index:]
            data = [data[i] for i in random.sample(range(len(data)),int(args.max))]
            print("new data size %s" %len(data))
            # TODO: read those lines with something like `j = json.loads(line)`
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            # TODO: append the result to 'allOutput'
            h = 0
            for line in data:
                j = json.loads(line)
                d = {key:value for (key, value) in j.items() if key in ('id', 'body')}
                d["cat"] = file
                d["body"] = preproc1(d["body"])
                allOutput.append(d)
                h+=1
                print(h)

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (int(args.max) > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    main(args)

