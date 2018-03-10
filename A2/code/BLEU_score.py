import math



def ref_gram_count(refs):
    ref_g_count = {}
    for ref in refs:
        #print("processing ref:{}".format(ref))
        words = ref.split()[1:-1]
        words_len = len(words)
        for i in range(len(words)):
            #print(1)
            first = words[i]
            if first not in ref_g_count.keys():
                ref_g_count[first] = {}
            #only one word in this sen
            if i+1 == words_len: continue
            second = words[i+1]
            if second not in ref_g_count[first].keys():
                ref_g_count[first][second] = {}
            if i + 2 == words_len: continue
            third = words[i+2]
            if third not in ref_g_count[first][second].keys():
                ref_g_count[first][second][third] = 1
            #print(2)
    return ref_g_count


def can_gram_count(ref_count, candidate):
    words = candidate.split()[1:-1]
    num_can_words = len(words)

    unigram_count = 0
    bigram_count = 0
    trigram_count = 0

    for i in range(len(words)):
        first = words[i]
        if first not in ref_count.keys():
            continue
        unigram_count += 1
        if i + 1 == num_can_words: continue

        sec = words[i + 1]
        if sec not in ref_count[first].keys():
            continue
        bigram_count += 1
        if i + 2 == num_can_words: continue

        third = words[i + 2]
        if third not in ref_count[first][sec].keys():
            continue
        trigram_count += 1
    return unigram_count,bigram_count,trigram_count

def br_penalty(candidate, references):
    words_len = len(candidate.split())
    nearest_len = float("inf")
    for ref in references:
        if abs(len(ref.split())-words_len) < abs(nearest_len-words_len):
            nearest_len= len(ref.split())
    brevity = (nearest_len-2)/(words_len-2)

    #len ref > len can
    if brevity <1:
        return 1
    return math.exp(1-brevity)


def BLEU_score(candidate, references, n):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing
	
	INPUTS:
	sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.

	
	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""
	
	#TODO: Implement by student.

    #grams in references
    gram_count_ref = ref_gram_count(references)
    #skip sentstart&end since it doesn't matter in this
    bleu_score =0
    words = candidate.split()[1:-1]
    num_can_words = len(words)

    unigram_count, bigram_count, trigram_count = can_gram_count(gram_count_ref, candidate)
    bp = br_penalty(candidate,references)

    uni_precision = unigram_count/num_can_words
    bi_precison = ((n>1)*bigram_count/(num_can_words-1)) + (n<=1)
    tri_precison = ((n>2)*trigram_count/(num_can_words-2)) + (n<=2)

    #print("bp:{}, uni_p:{}, bi_p:{}, tri_p:{}".format(bp,uni_precision,bi_precison,tri_precison))
    bleu_score = bp* ((uni_precision*bi_precison*tri_precison)**(1/n))

    return bleu_score