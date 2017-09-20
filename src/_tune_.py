import csv
import sys

import numpy as np
from sklearn import preprocessing as pr
from sklearn import svm
from sklearn.grid_search import GridSearchCV

import _config_constants_ as cons
import _config_globals_ as globals
import _feature_lexicon_score_ as lexicon_score
import _feature_micro_blog_score_ as micro_blog_score
import _feature_n_gram_ as ngram
import _feature_postag_ as postag
import _pre_process_ as ppros
import _writing_style_ as ws

TOTAL_TRAIN_DATA = 20633

POS_DATA = {}
NEG_DATA = {}
NEU_DATA = {}


def load_matrix_sub(process_dict, label):
    keys = process_dict.keys()
    vectors = []
    labels = []
    if len(keys) > 0:
        for key in keys:
            line = process_dict.get(key)
            z = map_tweet(line)
            vectors.append(z)
            labels.append(float(label))
    return vectors, labels


def map_tweet(tweet):
    """
    This function use to map the tweet
    :param tweet:
    :return:
    """

    vector = []

    preprocessed_tweet = ppros.pre_process_tweet(tweet)
    postag_tweet = postag.pos_tag_string(preprocessed_tweet)

    unigram_score = ngram.score(preprocessed_tweet, POS_UNI_GRAM, NEG_UNI_GRAM, NEU_UNI_GRAM, 1)
    post_unigram_score = ngram.score(postag_tweet, POS_POST_UNI_GRAM, NEG_POST_UNI_GRAM, NEU_POST_UNI_GRAM, 1)

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

    vector.append(afinn_score_96)
    vector.append(afinn_score_111)
    vector.append(lexicon_score_gen)
    vector.append(senti_140_score)
    vector.extend(NRC_score)
    vector.append(binliu_score)
    vector.append(sentiword_score)
    vector.extend(writing_style)
    vector.append(emoticon_score)
    vector.append(unicode_emoticon_score)
    vector.extend(post_unigram_score)
    vector.extend(unigram_score)
    return vector


with open("../dataset/semeval.csv", 'r') as main_dataset:
    main = csv.reader(main_dataset)
    pos_completed = neg_completed = neu_completed = False
    pos_count = neg_count = neu_count = 0
    for line in main:
        if line[1] == "positive" and pos_count < globals.POS_RATIO * TOTAL_TRAIN_DATA:
            POS_DATA.update({str(pos_count): line[2]})
            pos_count += 1
        elif line[1] == "positive" and pos_count > globals.POS_RATIO * TOTAL_TRAIN_DATA:
            pos_completed = True
        if line[1] == "negative" and neg_count < globals.NEG_RATIO * TOTAL_TRAIN_DATA:
            NEG_DATA.update({str(neg_count): line[2]})
            neg_count += 1
        elif line[1] == "negative" and neg_count > globals.NEG_RATIO * TOTAL_TRAIN_DATA:
            neg_completed = True
        if line[1] == "neutral" and neu_count < globals.NEU_RATIO * TOTAL_TRAIN_DATA:
            NEU_DATA.update({str(neu_count): line[2]})
            neu_count += 1
        elif line[1] == "neutral" and neu_count > globals.NEU_RATIO * TOTAL_TRAIN_DATA:
            neu_completed = True

        if pos_completed and \
                neg_completed and \
                neu_completed:
            break

POS_UNI_GRAM, POS_POST_UNI_GRAM = ngram.ngram(POS_DATA, 1)
NEG_UNI_GRAM, NEG_POST_UNI_GRAM = ngram.ngram(NEG_DATA, 1)
NEU_UNI_GRAM, NEU_POST_UNI_GRAM = ngram.ngram(NEU_DATA, 1)
pos_vec, pos_lab = load_matrix_sub(POS_DATA, 2.0)
neg_vec, neg_lab = load_matrix_sub(NEG_DATA, -2.0)
neu_vec, neu_lab = load_matrix_sub(NEU_DATA, 0.0)
vectors = pos_vec + neg_vec + neu_vec
labels = pos_lab + neg_lab + neu_lab
vectors_scaled = pr.scale(np.array(vectors))
scaler = pr.StandardScaler().fit(vectors)
vectors_normalized = pr.normalize(vectors_scaled, norm='l2')
normalizer = pr.Normalizer().fit(vectors_scaled)
vectors = vectors_normalized
vectors = vectors.tolist()


# These code block are directly added from
# https://github.com/Subangani/TSA-with-SelfTraining/blob/master/src/svmTune.py


def get_best_svm():
    parameters = {'kernel': ['linear', 'rbf'], 'C': [0.01 * i for i in range(1, 100, 1)],
                  'gamma': [0.01 * i for i in range(1, 4, 1)]}
    svr = svm.SVC(class_weight={2.0: 1.47, 0.0: 1.0, -2.0: 3.125})
    grid = GridSearchCV(svr, parameters, scoring='f1_weighted', n_jobs=-1, cv=10)
    tunes_model = grid.fit(vectors, labels)
    print tunes_model.best_params_, tunes_model.best_score_


#
# def get_best_xgboost():
#     cv_params = {'max_depth': [2*i+1 for i in range(1,4,1)],
#                  'min_child_weight': [2*i+1 for i in range(0, 3,1)]}
#     ind_params = {'learning_rate': 0.1, 'n_estimators': 50, 'seed':0, 'subsample': 0.8,
#                   'colsample_bytree': 0.8,'objective': 'multi:softmax','silent': 0,}
#     optimized_GBM = GridSearchCV(XGBClassifier(**ind_params).fit(np.asarray(vectors_train),labels_train),
#                                  param_grid=cv_params,scoring= 'accuracy', cv= 5, n_jobs = -1)
#     tuned_model=optimized_GBM.fit(np.asarray(vectors_tune), labels_tune)
#     print tuned_model
#     print tuned_model.best_params_,tuned_model.best_score_

if sys.argv[1] == cons.CLASSIFIER_SVM:
    print "Tuning SVM"
    get_best_svm()
elif sys.argv[1] == cons.CLASSIFIER_XGBOOST:
    print "Tuning XGBoost"
    # get_best_xgboost()
