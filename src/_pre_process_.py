import re


def load_stop_word_list():
    fp = open("../resource/stopWords.txt", 'r')
    stop_words = ['at_user', 'url']
    line = fp.readline()
    while line:
        word = line.strip()
        stop_words.append(word)
        line = fp.readline()
    fp.close()
    return stop_words


def remove_stop_words(tweet, stop_words):
    result = ''
    for w in tweet:
        if w[-4:] == "_NEG":
            if w[:-4] in stop_words:
                None
            else:
                result = result + w + ' '
        else:
            if w in stop_words:
                None
            else:
                result = result + w + ' '
    return result


def negate(tweets):
    fn = open("../resource/negation.txt", "r")
    line = fn.readline()
    negation_list = []
    while line:
        negation_list.append(line.split(None, 1)[0]);
        line = fn.readline()
    fn.close()
    punctuation_marks = [".", ":", ";", "!", "?"]
    break_words = ["but"]

    for i in range(len(tweets)):
        if tweets[i] in negation_list:
            j = i + 1
            while j < len(tweets):
                if tweets[j][-1] not in (punctuation_marks and break_words):
                    tweets[j] = tweets[j] + "_NEG"
                    j = j + 1
                elif tweets[j][-1] not in (punctuation_marks and break_words):
                    tweets[j] = tweets[j][-1] + "_NEG"
                else:
                    break
            i = j
    return tweets


def load_internet_slangs_list():
    fi = open('../resource/internetSlangs.txt', 'r')
    slangs = {}
    line = fi.readline()
    while line:
        l = line.split(r',%,')
        if len(l) == 2:
            slangs[l[0]] = l[1][:-2]
        line = fi.readline()
    fi.close()
    return slangs


def replace_slangs(tweet, slangs_list):
    result = ''
    words = tweet.split()
    for w in words:
        if w in slangs_list.keys():
            result = result + slangs_list[w] + " "
        else:
            result = result + w + " "
    return result


def replace_two_or_more(s):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


def pre_process_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'url', tweet)
    tweet = re.sub('((www\.[^\s]+)|(http?://[^\s]+))', 'url', tweet)
    tweet = re.sub('@[^\s]+', 'at_user', tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = tweet.strip('\'"')
    processed_tweet = replace_two_or_more(tweet)
    slangs = load_internet_slangs_list()
    words = replace_slangs(processed_tweet, slangs).split()
    negated_tweets = negate(words)
    stop_words = load_stop_word_list()
    preprocessed_tweet = remove_stop_words(negated_tweets, stop_words)
    punctuation_removed_tweets = re.sub('[^a-zA-Z_]+', ' ', preprocessed_tweet)
    final_processed_tweets = punctuation_removed_tweets.replace(" _NEG", "_NEG")
    return final_processed_tweets
