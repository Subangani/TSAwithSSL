import warnings
import _config_globals_ as globals
import _feature_lexicon_score_ as lexicon_score
import _feature_n_gram_ as ngram
import _feature_postag_ as postag
import _pre_process_ as ppros
import _writing_style_ as ws
import _feature_micro_blog_score_ as micro_blog_score
import _global_data_store_ as ds
warnings.filterwarnings('ignore')


def get_vectors_and_labels():
    ds.POS_UNI_GRAM, ds.POS_POST_UNI_GRAM = ngram.ngram(file_dict=ds.POS_DICT, gram=1)
    ds.NEG_UNI_GRAM, ds.NEG_POST_UNI_GRAM = ngram.ngram(file_dict=ds.NEG_DICT, gram=1)
    ds.NEU_UNI_GRAM, ds.NEU_POST_UNI_GRAM = ngram.ngram(file_dict=ds.NEU_DICT, gram=1)
    pos_vec, pos_lab, kpos = load_matrix_sub(process_dict=ds.POS_DICT,label=2.0, is_self_training=False)
    neg_vec, neg_lab, kneg = load_matrix_sub(process_dict=ds.NEG_DICT,label=-2.0, is_self_training=False)
    neu_vec, neu_lab, kneu = load_matrix_sub(process_dict=ds.NEU_DICT,label=0.0, is_self_training=False)
    ds.VECTORS = pos_vec + neg_vec + neu_vec
    ds.LABELS = pos_lab + neg_lab + neu_lab
    is_success = True
    return is_success


def load_matrix_sub(process_dict,label=0.0, is_self_training=False):
    """
    :param process_dict:
    :param label:
    :param random_list:
    :param is_randomized:
    :param is_self_training:
    :return:
    """
    limit_t = globals.LABEL_LIMIT
    limit_i = globals.INCREMENT_LIMIT
    if limit_i != 0 or limit_t != 0:
        keys = process_dict.keys()
        if len(keys) > 0:
            vectors = []
            labels = []
            k_pol = 0
            for key in keys:
                line = process_dict.get(key)
                k_pol += 1
                z = map_tweet(line,is_self_training)
                vectors.append(z)
                labels.append(float(label))
        else:
            vectors = []
            labels = []
            k_pol = 0
    else:
        vectors = []
        labels = []
        k_pol = 0
    return vectors, labels, k_pol


def map_tweet(tweet,is_self_training):
    """
    This function use to map the tweet
    :param tweet:
    :param is_self_training:
    :return:
    """

    feature_set_code = globals.FEATURE_SET_CODE

    vector = []

    preprocessed_tweet = ppros.pre_process_tweet(tweet)
    postag_tweet = postag.pos_tag_string(preprocessed_tweet)

# Score obtaining phase these are common for selftraining except obtaining unigram and
# postag unigram score

    if not is_self_training:
        unigram_score = ngram.score(preprocessed_tweet, ds.POS_UNI_GRAM, ds.NEG_UNI_GRAM, ds.NEU_UNI_GRAM, 1)
        post_unigram_score = ngram.score(postag_tweet, ds.POS_POST_UNI_GRAM, ds.NEG_POST_UNI_GRAM, ds.NEU_POST_UNI_GRAM, 1)
    else:
        unigram_score = ngram.score(preprocessed_tweet, ds.POS_UNI_GRAM_SELF, ds.NEG_UNI_GRAM_SELF, ds.NEU_UNI_GRAM_SELF, 1)
        post_unigram_score = ngram.score(postag_tweet, ds.POS_POST_UNI_GRAM_SELF, ds.NEG_POST_UNI_GRAM_SELF, ds.NEU_POST_UNI_GRAM_SELF, 1)

    lexicon_score_gen = lexicon_score.get_lexicon_score(preprocessed_tweet)
    afinn_score_96 = lexicon_score.get_afinn_99_score(preprocessed_tweet)
    afinn_score_111 = lexicon_score.get_afinn_111_score(preprocessed_tweet)
    senti_140_score = lexicon_score.get_senti140_score(preprocessed_tweet)
    NRC_score = lexicon_score.get_NRC_score(preprocessed_tweet)
    binliu_score = lexicon_score.get_senti_word_net_score(preprocessed_tweet)
    sentiword_score = lexicon_score.get_binliu_score(preprocessed_tweet)

    emoticon_score = micro_blog_score.emoticon_score(tweet)
    unicode_emoticon_score = micro_blog_score.unicode_emoticon_score(tweet)

    writing_style = ws.writing_style_vector(tweet)

# These classification are just for ease of division in general practice
# Generally we use default feature code 15 which takes all the feature
# You can evaluate that by analysing below code blocks :)

    if feature_set_code % 2 == 1:
        vector.append(afinn_score_96)
        vector.append(afinn_score_111)
        vector.append(lexicon_score_gen)
        vector.append(senti_140_score)
        vector.extend(NRC_score)
        vector.append(binliu_score)
        vector.append(sentiword_score)
    if feature_set_code % 4 >= 2:
        vector.extend(writing_style)
    if feature_set_code % 8 >= 4:
        vector.append(emoticon_score)
        vector.append(unicode_emoticon_score)
    if feature_set_code % 16 >= 8:
        vector.extend(post_unigram_score)
        vector.extend(unigram_score)
    return vector
