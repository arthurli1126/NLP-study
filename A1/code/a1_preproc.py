import sys
import argparse
import os
import json
import random
import html
import re
import string
import spacy

#indir = '/u/cs401/A1/data/';
#sw_path = 'u/cs401/Wordlists/StopWords';
dev_dir = './data/';
dev_sw_path ='../Wordlists/StopWords';

def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''

    modComm = ''
    if 1 in steps:
        comment = remove_newline(comment)
    if 2 in steps:
        comment = replace_html_code(comment)
    if 3 in steps:
        comment = remove_urls(comment)
    if 4 in steps:
        comment = split_punctuatuin(comment)
    if 5 in steps:
        pass
    if 6 in steps:
        comment = token_tag(comment)
    if 7 in steps:
        comment = remove_sd(comment)
    if 8 in steps:
        print('TODO')
    if 9 in steps:
        print('TODO')
    if 10 in steps:
        print('TODO')

    modComm = comment
    return modComm

def remove_newline(comment):
    return re.sub("\n"," ",comment)

def replace_html_code(comment):
    return html.unescape(comment)

def remove_urls(comment):
    pattern = r'http\S+'
    comment = re.sub(pattern, ' ', comment)
    pattern = r'www\S+'
    return re.sub(pattern, ' ', comment)

def split_punctuatuin(comment):
    punctuation = string.punctuation.replace("\'", "")
    comment = re.sub('[' + punctuation + ']', ' ', comment)
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

def token_tag(comment):
    modcom = ''
    nlp = spacy.load('en', disable=['parse','ner'])
    utt = nlp(comment)
    for token in utt:
        modcom = modcom + " " +token.text +"/" +token.tag_
    #AL-first character was space
    return modcom[1:]

def remove_sd(comment):
    st_words_file = open(dev_sw_path)
    st_words = st_words_file.readlines()
    st_words = [i.replace("\n","") for i in st_words]
    for wd in st_words:
        comment = re.sub(r'\b{}(\/\S*\b)?\b'.format(wd),'', comment)
    return comment












def main( args ):

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
            data = [data[i] for i in random.sample(range(len(data)),args.max)]
            print("new data size %s" %len(data))
            # TODO: read those lines with something like `j = json.loads(line)`
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            # TODO: append the result to 'allOutput'
            for line in data:
                j = json.loads(line)
                d = {key:value for (key, value) in j.items() if key in ('id', 'body')}
                d["cat"] = file
                #d["body"] = preproc1(d["body"])
                print(d["body"])
            exit(0)
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


    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    print(args)
    main(args)

