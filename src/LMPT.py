import warnings
import _config_globals_ as globals
import _feature_lexicon_score_ as lexicon_score
import _feature_n_gram_ as ngram
import _feature_postag_ as postag
import _pre_process_ as ppros
import _writing_style_ as ws
import _feature_micro_blog_score_ as micro_blog_score
import _global_data_store_ as ds
import _config_constants_ as cons
import _generic_commons_ as commons
from sklearn import preprocessing as pr, svm
from xgboost import XGBClassifier
import numpy as np
import csv
warnings.filterwarnings('ignore')


def load_initial_dictionaries(type):
    """
    This used to classify initial dataset as positive,negative and neutral
    :return: It return the success or failure
    """
    print globals.POS_COUNT_LIMIT, globals.NEG_COUNT_LIMIT, globals.NEU_COUNT_LIMIT
    pos_dict = {}
    neg_dict = {}
    neu_dict = {}
    unlabel_dict = {}
    if not type:
        with open('../dataset/unlabeled.csv', "r") as unlabeled_file:
            reader = csv.reader(unlabeled_file)
            count = 0
            for line in reader:
                count = count + 1
                tweet = line[5]
                if count < (10000):
                    unlabel_dict.update({str(count): tweet})
                else:
                    break
    else:
        with open("../dataset/semeval.csv", 'r') as main_dataset:
            main = csv.reader(main_dataset)
            pos_count = 1
            neg_count = 1
            neu_count = 1
            unlabel_count = 1
            for line in main:
                if line[1] == "positive" and pos_count <= globals.POS_COUNT_LIMIT:
                    pos_dict.update({str(pos_count): str(line[2])})
                    pos_count +=1
                elif line[1] == "negative" and neg_count <= globals.NEG_COUNT_LIMIT:
                    neg_dict.update({str(neg_count): str(line[2])})
                    neg_count +=1
                elif line[1] == "neutral" and neu_count <= globals.NEU_COUNT_LIMIT:
                    neu_dict.update({str(neu_count): str(line[2])})
                    neu_count +=1
                else:
                    if not type:
                        break
                    else:
                        unlabel_dict.update({str(unlabel_count): str(line[2])})
                        unlabel_count +=1
        ds.POS_DICT = pos_dict
        ds.NEG_DICT = neg_dict
        ds.NEU_DICT = neu_dict
        ds.UNLABELED_DICT = unlabel_dict
    return


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


def get_vectors_and_labels_self():
    """
    obtain the vectors and labels for total self training and storing it at main store
    :return:
    """
    pos_t, pos_post_t = ngram.ngram(ds.POS_DICT_SELF, 1)
    neg_t, neg_post_t = ngram.ngram(ds.NEG_DICT_SELF, 1)
    neu_t, neu_post_t = ngram.ngram(ds.NEU_DICT_SELF, 1)
    ds.POS_UNI_GRAM_SELF, is_success = commons.dict_update(ds.POS_UNI_GRAM, pos_t)
    ds.NEG_UNI_GRAM_SELF, is_success = commons.dict_update(ds.NEG_UNI_GRAM, neg_t)
    ds.NEU_UNI_GRAM_SELF, is_success = commons.dict_update(ds.NEU_UNI_GRAM, neu_t)
    ds.POS_POST_UNI_GRAM_SELF, is_success = commons.dict_update(ds.POS_POST_UNI_GRAM, pos_post_t)
    ds.NEG_POST_UNI_GRAM_SELF, is_success = commons.dict_update(ds.NEG_POST_UNI_GRAM, neg_post_t)
    ds.NEU_POST_UNI_GRAM_SELF, is_success = commons.dict_update(ds.NEU_POST_UNI_GRAM, neu_post_t)
    pos_vec, pos_lab, kpos = load_matrix_sub(ds.POS_DICT_SELF, 2.0,True)
    neg_vec, neg_lab, kneg = load_matrix_sub(ds.NEG_DICT_SELF, -2.0,True)
    neu_vec, neu_lab, kneu = load_matrix_sub(ds.NEU_DICT_SELF, 0.0,True)
    ds.VECTORS_SELF = ds.VECTORS + pos_vec + neg_vec + neu_vec
    ds.LABELS_SELF = ds.LABELS + pos_lab + neg_lab + neu_lab
    return is_success


def generate_model(is_self_training=False):
    """
    generating model and storing in main data store
    :param is_self_training:
    :return:
    """
    if not is_self_training:
        vectors = ds.VECTORS
        labels = ds.LABELS
    else:
        vectors = ds.VECTORS_SELF
        labels = ds.LABELS_SELF
    classifier_type = globals.DEFAULT_CLASSIFIER
    vectors_scaled = pr.scale(np.array(vectors))
    scaler = pr.StandardScaler().fit(vectors)
    vectors_normalized = pr.normalize(vectors_scaled, norm='l2')
    normalizer = pr.Normalizer().fit(vectors_scaled)
    vectors = vectors_normalized
    vectors = vectors.tolist()
    if classifier_type == cons.CLASSIFIER_SVM:
        kernel_function = globals.DEFAULT_KERNEL
        c_parameter = globals.DEFAULT_C_PARAMETER
        gamma = globals.DEFAULT_GAMMA_SVM
        class_weights = globals.DEFAULT_CLASS_WEIGHTS
        model = svm.SVC(kernel=kernel_function, C=c_parameter,
                        class_weight=class_weights, gamma=gamma,probability=True)
        model.fit(vectors, labels)
    elif classifier_type == cons.CLASSIFIER_XGBOOST:
        learning_rate = globals.DEFAULT_LEARNING_RATE
        max_depth = globals.DEFAULT_MAX_DEPTH
        min_child_weight = globals.DEFAULT_MIN_CHILD_WEIGHT
        silent = globals.DEFAULT_SILENT
        objective = globals.DEFAULT_OBJECTIVE
        subsample = globals.DEFAULT_SUB_SAMPLE
        gamma = globals.DEFAULT_GAMMA_XBOOST
        reg_alpha = globals.DEFAULT_REGRESSION_ALPHA
        n_estimators = globals.DEFAULT_N_ESTIMATORS
        colsample_bytree = globals.DEFAULT_COLSAMPLE_BYTREE
        model = XGBClassifier(learning_rate = learning_rate, max_depth = max_depth,
                              min_child_weight = min_child_weight,silent = silent,
                              objective = objective, subsample = subsample,gamma = gamma,
                              reg_alpha = reg_alpha, n_estimators = n_estimators,
                              colsample_bytree = colsample_bytree,)
        vectors_a = np.asarray(vectors)
        model.fit(vectors_a,labels)
    else:
        model = None
    ds.SCALAR = scaler
    ds.NORMALIZER = normalizer
    ds.MODEL = model
    return


def predict(tweet,is_self_training):
    z = map_tweet(tweet, is_self_training)
    z_scaled = ds.SCALAR.transform(z)
    z = ds.NORMALIZER.transform([z_scaled])
    z = z[0].tolist()
    return ds.MODEL.predict([z]).tolist()[0]


def predict_probability(tweet,is_self_training):
    z = map_tweet(tweet, is_self_training)
    z_scaled = ds.SCALAR.transform(z)
    z = ds.NORMALIZER.transform([z_scaled])
    z = z[0].tolist()
    return ds.MODEL.predict_proba([z]).tolist()[0]


def predict_probability_compare(nl,na):
    max_proba = max(na)
    if max > 0.5:
        if na[0] == max_proba and nl == -2.0:
            return True
        elif na[1] == max_proba and nl == 0.0:
            return True
        elif na[2] == max_proba and nl == 2.0:
            return True
        else:
            return False
    else:
        return False


def store_test(is_self_training):
    test_dict = {}
    limit = globals.TEST_LIMIT
    with open('../dataset/test.csv', "r") as testFile:
        reader = csv.reader(testFile)
        count = 0
        for line in reader:
            line = list(line)
            tweet = line[2]
            s = line[1]
            nl = predict(tweet,is_self_training)
            test_dict.update({str(count): [s, tweet, nl]})
            count = count + 1
            if count >= limit:
                break
    ds.TEST_DICT = test_dict
    return


def load_iteration_dict(is_self_training):
    """
    divide the unlabelled data to do self training
    :param is_self_training:
    :return:
    """
    if len(ds.UNLABELED_DICT) > 0:
        increment_limit = globals.INCREMENT_LIMIT
        pos_count = 0
        neg_count = 0
        neu_count = 0
        temp_pos_dict = {}
        temp_neg_dict = {}
        temp_neu_dict = {}
        pos_finished = False
        neg_finished = False
        neu_finished = False
        for key in ds.UNLABELED_DICT.keys():
            tweet = ds.UNLABELED_DICT.get(key)
            nl = predict(tweet,is_self_training)
            na = predict_probability(tweet,is_self_training)
            is_success = predict_probability_compare(nl,na)
            if is_success :
                if nl == 2.0 and pos_count < globals.POS_RATIO * increment_limit:
                    temp_pos_dict[str(pos_count)] = tweet
                    pos_count = pos_count + 1
                    del ds.UNLABELED_DICT[key]
                elif nl == 2.0 and pos_count >= globals.POS_RATIO * increment_limit:
                    pos_finished = True

                if nl == -2.0 and neg_count < globals.NEG_RATIO * increment_limit:
                    temp_neg_dict[str(neg_count)] = tweet
                    neg_count = neg_count + 1
                    del ds.UNLABELED_DICT[key]
                elif nl == -2.0 and neg_count >= globals.NEG_RATIO * increment_limit:
                    neg_finished = True

                if nl == 0 and neu_count < globals.NEU_RATIO * increment_limit:
                    temp_neu_dict[str(neu_count)] = tweet
                    neu_count = neu_count + 1
                    del ds.UNLABELED_DICT[key]
                elif nl == 0 and neu_count >= globals.NEU_RATIO * increment_limit:
                    neu_finished = True

            if pos_finished and neg_finished and neu_finished :
                break
    else:
        temp_pos_dict = {}
        temp_neg_dict = {}
        temp_neu_dict = {}
    ds.POS_DICT_SELF = temp_pos_dict
    ds.NEG_DICT_SELF = temp_neg_dict
    ds.NEU_DICT_SELF = temp_neu_dict
    return