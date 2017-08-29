from sklearn import preprocessing as pr, svm
from xgboost import XGBClassifier
import csv
import os
import time
import numpy as np
import _generic_commons_ as commons
import _config_globals_ as globals
import _config_constants_ as cons
import classifier


def self_training():
    """
    Main Program
    :return:
    """
    directory = "../dataset/analysed/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    if os.path.exists(directory):
        feature_set_code = globals.FEATURE_SET_CODE
        label_limit = globals.LABEL_LIMIT
        test_limit = globals.TEST_LIMIT
        increment_limit = globals.INCREMENT_LIMIT
        no_of_iteration = globals.NO_OF_ITERATION
        classifier_type = globals.DEFAULT_CLASSIFIER

        file_prefix = str(label_limit) + "_" + str(feature_set_code) + "_" + \
                      str(test_limit) + "_" + str(increment_limit) + "_" + \
                      str(no_of_iteration) + "_" + str(classifier_type) + "_"

        with open('../dataset/analysed/' + file_prefix + 'result.csv', 'w') as result:

            csv_result = csv.writer(result)
            csv_result.writerow(globals.CSV_HEADER)

            time_list = [time.time()]

            classifier.ds.POS_DICT, classifier.ds.NEG_DICT, \
            classifier.ds.NEU_DICT, classifier.ds.UNLABELED_DICT = \
                commons.generate_dictionaries(1)
            print "Generated Initial Dictionaries"

            pos_len = len(classifier.ds.POS_DICT)
            neg_len = len(classifier.ds.NEG_DICT)
            neu_len = len(classifier.ds.NEU_DICT)

            classifier.get_vectors_and_labels()
            print "Obtained Vectors and Labels"
            generate_model(is_self_training=False)
            print "Generated Model"
            store_test(is_self_training=False)
            print "Stored Test"

            test_len = len(classifier.ds.TEST_DICT)

            print \
                (
                    pos_len, neg_len, neu_len, feature_set_code, test_len, 0
                )\
                , commons.get_score(classifier.ds.TEST_DICT)

            csv_result.writerow(
                (
                    pos_len, neg_len, neu_len, feature_set_code, test_len, 0
                )
                +
                commons.get_score(classifier.ds.TEST_DICT)
            )

            fscore_best = commons.get_score(classifier.ds.TEST_DICT)[9]
            successful_addition = 0
            i = 1
            while i < no_of_iteration:
                if i == 1:
                    is_self_training = False
                else:
                    is_self_training = True

                create_iteration_dict(is_self_training)
                print "Iteration " + str(i)
                print "Generated Dictionaries of the Iteration"

                pos_len = pos_len + len(classifier.ds.POS_DICT_SELF)
                neg_len = neg_len + len(classifier.ds.NEG_DICT_SELF)
                neu_len = neu_len + len(classifier.ds.NEU_DICT_SELF)
                pos_len_self = pos_len
                neg_len_self = neg_len
                neu_len_self = neu_len

                get_vectors_and_labels_self()
                print "Obtained Vectors and Labels"
                generate_model(is_self_training=True)
                print "Generated Model"
                store_test(is_self_training=True)
                print "Stored Test"

                print \
                    (
                        pos_len_self, neg_len_self, neu_len_self, feature_set_code, test_len, i
                    ) \
                    , \
                    commons.get_score(classifier.ds.TEST_DICT)

                csv_result.writerow(
                    (
                        pos_len_self, neg_len_self, neu_len_self, feature_set_code, test_len, i
                    )
                    +
                    commons.get_score(classifier.ds.TEST_DICT)
                )

                fscore_iter = commons.get_score(classifier.ds.TEST_DICT)[9]

                if fscore_best < fscore_iter:
                    classifier.ds.POS_UNI_GRAM = classifier.ds.POS_UNI_GRAM_SELF
                    classifier.ds.NEG_UNI_GRAM = classifier.ds.NEG_UNI_GRAM_SELF
                    classifier.ds.NEU_UNI_GRAM = classifier.ds.NEU_UNI_GRAM_SELF
                    classifier.ds.POS_POST_UNI_GRAM = classifier.ds.POS_POST_UNI_GRAM_SELF
                    classifier.ds.NEG_POST_UNI_GRAM = classifier.ds.NEG_POST_UNI_GRAM_SELF
                    classifier.ds.NEU_POST_UNI_GRAM = classifier.ds.NEU_POST_UNI_GRAM_SELF
                    classifier.ds.VECTORS = classifier.ds.VECTORS_SELF
                    classifier.ds.LABELS = classifier.ds.LABELS_SELF
                    fscore_best = fscore_iter
                    successful_addition += 1
                else:
                    for key in classifier.ds.POS_DICT_SELF.keys():
                        classifier.ds.UNLABELED_DICT.update({str(cons.DATA_SET_SIZE * (i + 1) + int(key)): classifier.ds.POS_DICT_SELF.get(key)})
                    for key in classifier.ds.NEG_DICT_SELF.keys():
                        classifier.ds.UNLABELED_DICT.update({str(cons.DATA_SET_SIZE * (i + 1) + int(key)): classifier.ds.NEG_DICT_SELF.get(key)})
                    for key in classifier.ds.NEU_DICT_SELF.keys():
                        classifier.ds.UNLABELED_DICT.update({str(cons.DATA_SET_SIZE * (i + 1) + int(key)): classifier.ds.NEU_DICT_SELF.get(key)})
                i = i + 1
            time_list.append(time.time())
            print successful_addition
            print commons.temp_difference_cal(time_list)
    else:
        print "Error in access directory ", directory
    return


def get_vectors_and_labels_self():
    """
    obtain the vectors and labels for total self training and storing it at main store
    :return:
    """
    pos_t, pos_post_t = classifier.ngram.ngram(classifier.ds.POS_DICT_SELF, 1)
    neg_t, neg_post_t = classifier.ngram.ngram(classifier.ds.NEG_DICT_SELF, 1)
    neu_t, neu_post_t = classifier.ngram.ngram(classifier.ds.NEU_DICT_SELF, 1)
    classifier.ds.POS_UNI_GRAM_SELF, is_success = commons.dict_update(classifier.ds.POS_UNI_GRAM, pos_t)
    classifier.ds.NEG_UNI_GRAM_SELF, is_success = commons.dict_update(classifier.ds.NEG_UNI_GRAM, neg_t)
    classifier.ds.NEU_UNI_GRAM_SELF, is_success = commons.dict_update(classifier.ds.NEU_UNI_GRAM, neu_t)
    classifier.ds.POS_POST_UNI_GRAM_SELF, is_success = commons.dict_update(classifier.ds.POS_POST_UNI_GRAM, pos_post_t)
    classifier.ds.NEG_POST_UNI_GRAM_SELF, is_success = commons.dict_update(classifier.ds.NEG_POST_UNI_GRAM, neg_post_t)
    classifier.ds.NEU_POST_UNI_GRAM_SELF, is_success = commons.dict_update(classifier.ds.NEU_POST_UNI_GRAM, neu_post_t)
    pos_vec, pos_lab, kpos = classifier.load_matrix_sub(classifier.ds.POS_DICT_SELF, 2.0,True)
    neg_vec, neg_lab, kneg = classifier.load_matrix_sub(classifier.ds.NEG_DICT_SELF, -2.0,True)
    neu_vec, neu_lab, kneu = classifier.load_matrix_sub(classifier.ds.NEU_DICT_SELF, 0.0,True)
    classifier.ds.VECTORS_SELF = classifier.ds.VECTORS + pos_vec + neg_vec + neu_vec
    classifier.ds.LABELS_SELF = classifier.ds.LABELS + pos_lab + neg_lab + neu_lab
    return is_success


def generate_model(is_self_training=False):
    """
    generating model and storing in main data store
    :param is_self_training:
    :return:
    """
    if not is_self_training:
        vectors = classifier.ds.VECTORS
        labels = classifier.ds.LABELS
    else:
        vectors = classifier.ds.VECTORS_SELF
        labels = classifier.ds.LABELS_SELF
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
    classifier.ds.SCALAR = scaler
    classifier.ds.NORMALIZER = normalizer
    classifier.ds.MODEL = model
    return


def predict(tweet,is_self_training):
    z = classifier.map_tweet(tweet, is_self_training)
    z_scaled = classifier.ds.SCALAR.transform(z)
    z = classifier.ds.NORMALIZER.transform([z_scaled])
    z = z[0].tolist()
    return classifier.ds.MODEL.predict([z]).tolist()[0]


def predict_probability(tweet,is_self_training):
    z = classifier.map_tweet(tweet, is_self_training)
    z_scaled = classifier.ds.SCALAR.transform(z)
    z = classifier.ds.NORMALIZER.transform([z_scaled])
    z = z[0].tolist()
    return classifier.ds.MODEL.predict_proba([z]).tolist()[0]


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
    classifier.ds.TEST_DICT = test_dict
    return


def create_iteration_dict(is_self_training):
    """
    divide the unlabelled data to do self training
    :param is_self_training:
    :return:
    """
    if len(classifier.ds.UNLABELED_DICT) > 0:
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
        for key in classifier.ds.UNLABELED_DICT.keys():
            tweet = classifier.ds.UNLABELED_DICT.get(key)
            nl = predict(tweet,is_self_training)
            na = predict_probability(tweet,is_self_training)
            is_success = predict_probability_compare(nl,na)
            if is_success :
                if nl == 2.0 and pos_count < globals.POS_RATIO * increment_limit:
                    temp_pos_dict[str(pos_count)] = tweet
                    pos_count = pos_count + 1
                    del classifier.ds.UNLABELED_DICT[key]
                elif nl == 2.0 and pos_count >= globals.POS_RATIO * increment_limit:
                    pos_finished = True

                if nl == -2.0 and neg_count < globals.NEG_RATIO * increment_limit:
                    temp_neg_dict[str(neg_count)] = tweet
                    neg_count = neg_count + 1
                    del classifier.ds.UNLABELED_DICT[key]
                elif nl == -2.0 and neg_count >= globals.NEG_RATIO * increment_limit:
                    neg_finished = True

                if nl == 0 and neu_count < globals.NEU_RATIO * increment_limit:
                    temp_neu_dict[str(neu_count)] = tweet
                    neu_count = neu_count + 1
                    del classifier.ds.UNLABELED_DICT[key]
                elif nl == 0 and neu_count >= globals.NEU_RATIO * increment_limit:
                    neu_finished = True

            if pos_finished and neg_finished and neu_finished :
                break
    else:
        temp_pos_dict = {}
        temp_neg_dict = {}
        temp_neu_dict = {}
    classifier.ds.POS_DICT_SELF = temp_pos_dict
    classifier.ds.NEG_DICT_SELF = temp_neg_dict
    classifier.ds.NEU_DICT_SELF = temp_neu_dict
    return


self_training()