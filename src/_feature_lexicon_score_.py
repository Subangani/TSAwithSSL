from __future__ import division
import _feature_lexicon_dict_load_ as dict_load
import nltk


def get_lexicon_score(tweet):
    score = 0
    for w in tweet.split():
        if w in dict_load.POS_LEXICON_FILE:
            score += 1
        if w in dict_load.NEG_LEXICON_FILE:
            score -= 1
        if w.endswith("_NEG"):
            if len(w) > 4:
                if (w[0:len(w) - 4]) in dict_load.POS_LEXICON_FILE:
                    score -= 1
                if (w[0:len(w) - 4]) in dict_load.NEG_LEXICON_FILE:
                    score += 1
    return score


def get_afinn_99_score(tweet):
    p = 0.0
    nbr = 0
    for w in tweet.split():
        if w in dict_load.AFFIN_LOAD_96.keys():
            nbr += 1
            p += dict_load.AFFIN_LOAD_96[w]
    if nbr != 0:
        return p / nbr
    else:
        return 0.0


def get_afinn_111_score(tweet):
    p = 0.0
    nbr = 0
    for w in tweet.split():
        if w in dict_load.AFFIN_LOAD_111.keys():
            nbr += 1
            p += dict_load.AFFIN_LOAD_111[w]
    if nbr != 0:
        return p / nbr
    else:
        return 0.0

def get_senti140_score(tweet):
    words = tweet.split()
    unigram_list = get_ngram_word(words,1)
    uni_score=0.0
    for word in unigram_list:
        if dict_load.SENTI_140_UNIGRAM_DICT.has_key(word):
            uni_score+=float(dict_load.SENTI_140_UNIGRAM_DICT.get(word))
    return uni_score


def get_NRC_score(tweet):
    words = tweet.split()
    unigram_list = get_ngram_word(words,1)
    uni_score = 0.0
    hash_score = 0.0
    for word in unigram_list:
        if dict_load.NRC_UNIGRAM_DICT.has_key(word):
            uni_score += float(dict_load.NRC_UNIGRAM_DICT.get(word))
        if list(word)[0] == '#':
            ar=list(word)
            ar.remove("#")
            word = ''.join(ar)
            if not dict_load.NRC_HASHTAG_DICT.get(word) is None:
                if dict_load.NRC_HASHTAG_DICT.get(word) == 'positive\n':
                    hash_score += 1.0
                elif dict_load.NRC_HASHTAG_DICT.get(word) == 'negative\n':
                    hash_score -= 1.0
    return uni_score,hash_score


def get_ngram_word(words,gram):
    ngram_list = []
    for i in range(len(words) + 1 - gram):
        temp = ""
        if not words[i:i + gram] is "":
            if gram == 1:
                temp = words[i]
            elif gram == 2:
                temp = words[i] + " " + words[i + 1]
        ngram_list.append(temp)
    return ngram_list


def get_senti_word_net_score(tweet):
    try:
        nlpos = {'a': ['JJ', 'JJR', 'JJS'],
                 'n': ['NN', 'NNS', 'NNP', 'NNPS'],
                 'v': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'IN'],
                 'r': ['RB', 'RBR', 'RBS']}
        text = tweet.split()
        tags = nltk.pos_tag(text)
        tagged_tweets=[]
        for i in range(0,len(tags)):
            if tags[i][1] in nlpos['a']:
                tagged_tweets.append(tags[i][0]+"#a")
            elif tags[i][1] in nlpos['n']:
                tagged_tweets.append(tags[i][0]+"#n")
            elif tags[i][1] in nlpos['v']:
                tagged_tweets.append(tags[i][0]+"#v")
            elif tags[i][1] in nlpos['r']:
                tagged_tweets.append(tags[i][0]+"#r")
        score = 0.0
        for i in range(0,len(tagged_tweets)):
            if tagged_tweets[i] in dict_load.SENTI_WORD_NET_DICT:
                score += float(dict_load.SENTI_WORD_NET_DICT.get(tagged_tweets[i]))
        return score
    except:
        return None


def get_binliu_score(tweet):
    score=0.0
    for word in tweet.split():
        if dict_load.BING_LIU_DICT.has_key(word):
            if dict_load.BING_LIU_DICT.get(word)=='positive\n':
                score += 1
            if dict_load.BING_LIU_DICT.get(word)=='negative\n':
                score -= 1
    return score