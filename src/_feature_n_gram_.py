import _feature_postag_ as postag
import _generic_commons_ as commons


def create_dict(words, gram):
    """
    This is to obtain the create_dict of word of particular line
    :param words: 
    :param gram: 
    :return: give a create_dict of word(s) with proper format based generate_n_gram_dict values such as 1,2,3
    """
    temp_dict = {}
    for i in range(len(words) - gram):
        if not words[i:i + gram] is "":
            temp = ""
            if gram == 1:
                temp = words[i]
            elif gram == 2:
                temp = words[i] + " " + words[i + 1]
            elif gram == 3:
                temp = words[i] + " " + words[i + 1] + " " + words[i + 2]
            local_temp_value = temp_dict.get(temp)
            if local_temp_value is None:
                temp_dict.update({temp: 1})
            else:
                temp_dict.update({temp: local_temp_value + 1})
    return temp_dict


def generate_n_gram_dict(file_dict, gram):
    """
    this will return n-gram set for uni-gram,bi-gram and tri-gram, with frequency calculated
    for normal text and POS-tagged.
    :param file_dict:
    :param gram:
    :return: frequency dictionaries
    """
    word_freq_dict = {}
    postag_freq_dict = {}
    keys = file_dict.keys()
    for line_key in keys:
        try:
            line = file_dict.get(line_key)
            words = line.split()
            word_dict = create_dict(words, gram)
            word_freq_dict, is_success = commons.dict_update(word_freq_dict, word_dict)
            temp_postags = postag.pos_tag_string(line).split()
            if temp_postags != "":
                postags = temp_postags
                postag_dict = create_dict(postags, gram)
                postag_freq_dict, is_success = commons.dict_update(postag_freq_dict, postag_dict)
        except IndexError:
            print "Error"
    return word_freq_dict, postag_freq_dict


def score(tweet, p, n, ne, ngram):
    """
    This will find individual score of each word with respect to its polarity
    :param tweet: 
    :param p: 
    :param n: 
    :param ne: 
    :param ngram: 
    :return: return positive, negative, and neutral score
    """
    pos = 0
    neg = 0
    neu = 0
    dictof_grams = {}
    tweet_list = tweet.split()
    dictof_grams.update(create_dict(tweet_list, ngram))
    for element in dictof_grams.keys():
        posCount = float(get_count(element, p))
        negCount = float(get_count(element, n))
        neuCount = float(get_count(element, ne))
        totalCount = posCount + negCount + neuCount
        if totalCount != 0:
            pos += posCount / totalCount
            neg += negCount / totalCount
            neu += neuCount / totalCount
    return [pos, neg, neu]


def get_count(gram, pol):
    """
    This will count the positive,negative, and neutral count based on relevant dictionary present
    :param gram: 
    :param pol: 
    :return: return the availability of particular generate_n_gram_dict
    """
    count = 0.0
    try:
        temp = float(pol.get(gram))
        if temp > 0.0:
            count = temp
    except:
        TypeError
    return count
