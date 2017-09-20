import _feature_micro_blog_dict_load_ as dict_load


def emoticon_score(tweet):
    s = 0.0
    l = tweet.split(" ")
    nbr = 0
    for i in range(0, len(l)):
        if l[i] in dict_load.EMOTICON_DICT.keys():
            nbr = nbr + 1
            s = s + dict_load.EMOTICON_DICT[l[i]]
    if nbr != 0:
        s = s / nbr
    return s


def unicode_emoticon_score(tweet):
    s = 0.0
    nbr = 0
    tweet = tweet.split();
    for i in range(len(tweet)):
        old = tweet[i]
        new = old.replace("\U000", "0x")
        if new in dict_load.UNICODE_EMOTICON_DICT.keys():
            nbr = nbr + 1
            s = s + dict_load.UNICODE_EMOTICON_DICT[new]
    if nbr != 0:
        s = s / nbr
    return s
