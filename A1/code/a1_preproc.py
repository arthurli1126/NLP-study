import argparse
import html
import json
import os
import random
import re
import string
import sys
import spacy
from multiprocessing.pool import ThreadPool
import time

# indir = '/u/cs401/A1/data/';
# sw_path = '/u/cs401/Wordlists/StopWords';
# ab_path = '/u/cs401/Wordlists/abbrev.english';
dev_dir = '../data/';
dev_sw_path = '../Wordlists/StopWords';
dev_ab_path = '../Wordlists/abbrev.english';

# load abbrevs file
abbrev_file = open(dev_ab_path)
#abbrev_file = open(ab_path)
abbrevs = abbrev_file.readlines()
abbrev_file.close()
abbrevs = [i.replace("\n", "") for i in abbrevs]
# load st_words file
st_words_file = open(dev_sw_path)
#st_words_file = open(sw_path)
st_words = st_words_file.readlines()
st_words_file.close()
st_words = [i.replace("\n", "") for i in st_words]

# create nlp
nlp = spacy.load('en', disable=['parse', 'ner'])
punctuation = re.sub(r"[']", '', string.punctuation)

#regex parrtern
url_parrtern = re.compile(r'(http[^\s]*|www[^\s]*)',flags=re.I)
pun_parrten_1 =re.compile(r"(\b)(" + "|".join(abbrevs) + r")")
pun_parrten_2 = re.compile(r"(\w+)([{}]+)(\s+|$|\w+)".format(punctuation))


# TODO： check end of line step
def preproc1(comment, steps=range(1, 11)):
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
        comment = split_punctuation(comment)
    if 5 in steps:
        comment = split_clitics(comment)
    if 6 in steps:
        modComm_6, modComm_8 = spacy_support(comment)
        comment = modComm_6
    if 7 in steps:
        if 8 not in steps:
            comment = remove_sd(comment, st_words)
    if 8 in steps:
        if 6 in steps:
            comment = modComm_8
            if 7 in steps:
                comment = remove_sd(modComm_8, st_words)
        else:
            modComm_6, modComm_8 = spacy_support(comment)
            if 7 in steps:
                comment = remove_sd(modComm_8, st_words)
            else:
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
    return re.sub("\n", " ", comment)


'''
replace hmtl code with ascii
'''


def replace_html_code(comment):
    return html.unescape(comment)

'''
remove urls
'''
def remove_urls(comment):
    comment = re.sub(url_parrtern, '', comment)
    return comment




'''
split_punctuation
r"(\b)(" + "|".join(abbrevs) + r")"
r"(\w+)([{}]+)(\s+|$|\w+)".format(punctuation)
'''

def split_punctuation(comment):
    # change the period of abbrev to magic=xeq this is bad
    comment = re.sub(pun_parrten_1,
                     lambda m: m.group(1) + m.group(2).replace(".", "xeqxeq"), comment)
    comment = re.sub(pun_parrten_2, '\g<1> \g<2> \g<3>', comment).replace("xeqxeq", ".")
    comment = re.sub(r"\s+", ' ', comment)
    return comment


def split_clitics(comment):
    # clitics list will be sub except ' s' ' will need special way to separate it
    c_list = ['\'s',
              '\'re',
              '\'ve',
              '\'d',
              'n\'t',
              '\'ll',
              '\'m',
              '\'em',
              'y\'',
              'Y\'']
    pattern = re.compile(r'\b(\w*)(' + r"|".join(c_list) + r")(\w+|\s+)\b",flags=re.I)
    comment = re.sub(pattern,
                     lambda m: m.group(1) + " " + m.group(2) + " " +m.group(3), comment)

    pattern = re.compile(r"\b(\w+)(s\')(\w+|\s+)\b", flags=re.I)
    comment = re.sub(pattern,
                     lambda m: m.group(1)+m.group(2)[0] + " "+m.group(2)[1] + m.group(3), comment)
    comment = re.sub(r"\s+", ' ', comment)
    return comment


'''
step6&8 spacy support 
'''


def spacy_support(comment):
    modcom_6 = ''
    modcom_8 = ''
    utt = nlp(comment)
    # to-do AL if token start with - dont lemmanize it
    for token in utt:
        modcom_6 = modcom_6 + " " + token.text + "/" + token.tag_
        # modcom_8 = modcom_8 + " " + token.lemma_ + "/" + token.tag_
        modcom_8 = modcom_8 + " " + (token.lemma_ if token.lemma_[0] != '-' else token.text) + "/" + token.tag_
    # AL-first character was space
    return modcom_6[1:], modcom_8[1:]


def remove_sd(comment, st_words):
    pattern = re.compile(r"\b(" + r"|".join(st_words) + r")(\/\S*)?\b",flags=re.I)
    comment = re.sub(pattern, '', comment + ' ')
    comment = re.sub(r"\s+", ' ', comment)
    return comment


def add_newline_eos(comment):
    eos = ".?!"
    return re.sub(r"(\s+)([{}]+/.+\s|[{}]+/.+$)".format(eos,eos), '\g<1>\g<2>\n', comment)


# this one is little tricky
def convert_lc(comment):
    return re.sub(r"(\b\S+)(/\S+\b)", lambda m: m.group(1).lower() + m.group(2), comment)


'''
Going to implement multithreading 
for this module to speed up the process
'''


def process_wrapper(data, file, steps=range(1, 11)):
    comments = []
    # TODO: need to remove this in the furture
    h = 0
    for line in data:
        j = json.loads(line)
        d = {key: value for (key, value) in j.items() if key in ('id', 'body')}
        d["cat"] = file
        d["body"] = preproc1(d["body"], steps)
        comments.append(d)
        h += 1
        print("{}:{}".format(file,h))
    return comments


def main(args):
    allOutput = []
    thread_pool = ThreadPool(4)

    for subdir, dirs, files in os.walk(dev_dir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            start_index = args.ID[0] % len(data)
            print("start index : %s" % start_index)
            data = data[start_index:]
            data = [data[i] for i in random.sample(range(len(data)), int(args.max))]
            print("new data size %s" % len(data))
            #using 10 threads for each file
            for p in range(0, len(data), 1000):
                allOutput.append(thread_pool.apply_async(process_wrapper, (data[p:p+1000],file)))
    results=[]
    for r in allOutput:
        results += r.get()

    print("dang")
    fout = open(args.output, 'w')
    fout.write(json.dumps(results))
    fout.close()
    thread_pool.close()
    thread_pool.join()


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
    t_0 = time.time()
    main(args)
    print(time.time()-t_0)
