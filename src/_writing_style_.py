from __future__ import division


def capitalized_words_in_tweet(tweet):
    count = 0
    if len(tweet) != 0:
        for w in tweet.split():
            if w.isupper():
                if len(w) > 1:
                    count = count + 1
    return count


def exclamation_count(tweet):
    tweet_words = tweet.split()
    count = 0
    for word in tweet_words:
        if word.count("!") > 2:
            count += 1
    return count


def question_mark_count(tweet):
    tweet_words = tweet.split()
    count = 0
    for word in tweet_words:
        if word.count("?") > 2:
            count += 1
    return count


def capital_count_in_a_word(tweet):
    count = 0
    if len(tweet) != 0:
        for c in tweet:
            if str(c).isupper():
                count = count + 1
    return count


def surround_by_signs(tweet):
    """
    This is used to find the surrounding signs
    [based on AVAYA paper I have added this just to check whether there is improvement or not]
    :param tweet:
    :return:
    """
    highlight = ['"',"'","*"]
    count = 0
    if len(tweet) != 0:
        for c in tweet:
            if c[0] == c[len(c)-1] and c[0] in highlight:
                count = count + 1 ;
    return count


def writing_style_vector(tweet):
    cap_word = capitalized_words_in_tweet(tweet)
    exc_count = exclamation_count(tweet)
    que_count = question_mark_count(tweet)
    cap_count = capital_count_in_a_word(tweet)
    surr_count = surround_by_signs(tweet)
    return [cap_word, exc_count, que_count, cap_count,surr_count]

