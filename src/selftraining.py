import csv
import os
import time
import _generic_commons_ as commons
import _config_globals_ as globals
import _config_constants_ as cons
import LMPT


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

            LMPT.load_initial_dictionaries(1)
            LMPT.get_vectors_and_labels()
            LMPT.generate_model(is_self_training=False)
            LMPT.store_test(is_self_training=False)

            pos_len = len(LMPT.ds.POS_DICT)
            neg_len = len(LMPT.ds.NEG_DICT)
            neu_len = len(LMPT.ds.NEU_DICT)
            test_len = len(LMPT.ds.TEST_DICT)

            print \
                (
                    pos_len, neg_len, neu_len, feature_set_code, test_len, 0
                )\
                , commons.get_score(LMPT.ds.TEST_DICT)

            csv_result.writerow(
                (
                    pos_len, neg_len, neu_len, feature_set_code, test_len, 0
                )
                +
                commons.get_score(LMPT.ds.TEST_DICT)
            )

            fscore_best = commons.get_score(LMPT.ds.TEST_DICT)[9]
            successful_addition = 0
            i = 1
            while i < no_of_iteration:
                if i == 1:
                    is_self_training = False
                else:
                    is_self_training = True

                LMPT.load_iteration_dict(is_self_training)
                LMPT.get_vectors_and_labels_self()
                LMPT.generate_model(is_self_training=True)
                LMPT.store_test(is_self_training=True)

                pos_len = pos_len + len(LMPT.ds.POS_DICT_SELF)
                neg_len = neg_len + len(LMPT.ds.NEG_DICT_SELF)
                neu_len = neu_len + len(LMPT.ds.NEU_DICT_SELF)
                pos_len_self = pos_len
                neg_len_self = neg_len
                neu_len_self = neu_len

                print \
                    (
                        pos_len_self, neg_len_self, neu_len_self, feature_set_code, test_len, i
                    ) \
                    , \
                    commons.get_score(LMPT.ds.TEST_DICT)

                csv_result.writerow(
                    (
                        pos_len_self, neg_len_self, neu_len_self, feature_set_code, test_len, i
                    )
                    +
                    commons.get_score(LMPT.ds.TEST_DICT)
                )

                fscore_iter = commons.get_score(LMPT.ds.TEST_DICT)[9]

                if fscore_best < fscore_iter:
                    LMPT.ds.POS_UNI_GRAM = LMPT.ds.POS_UNI_GRAM_SELF
                    LMPT.ds.NEG_UNI_GRAM = LMPT.ds.NEG_UNI_GRAM_SELF
                    LMPT.ds.NEU_UNI_GRAM = LMPT.ds.NEU_UNI_GRAM_SELF
                    LMPT.ds.POS_POST_UNI_GRAM = LMPT.ds.POS_POST_UNI_GRAM_SELF
                    LMPT.ds.NEG_POST_UNI_GRAM = LMPT.ds.NEG_POST_UNI_GRAM_SELF
                    LMPT.ds.NEU_POST_UNI_GRAM = LMPT.ds.NEU_POST_UNI_GRAM_SELF
                    LMPT.ds.VECTORS = LMPT.ds.VECTORS_SELF
                    LMPT.ds.LABELS = LMPT.ds.LABELS_SELF
                    fscore_best = fscore_iter
                    successful_addition += 1
                else:
                    for key in LMPT.ds.POS_DICT_SELF.keys():
                        LMPT.ds.UNLABELED_DICT.update({str(cons.DATA_SET_SIZE * (i + 1) + int(key)): LMPT.ds.POS_DICT_SELF.get(key)})
                    for key in LMPT.ds.NEG_DICT_SELF.keys():
                        LMPT.ds.UNLABELED_DICT.update({str(cons.DATA_SET_SIZE * (i + 1) + int(key)): LMPT.ds.NEG_DICT_SELF.get(key)})
                    for key in LMPT.ds.NEU_DICT_SELF.keys():
                        LMPT.ds.UNLABELED_DICT.update({str(cons.DATA_SET_SIZE * (i + 1) + int(key)): LMPT.ds.NEU_DICT_SELF.get(key)})
                i = i + 1
            time_list.append(time.time())
            print successful_addition
            print commons.temp_difference_cal(time_list)
    else:
        print "Error in access directory ", directory
    return


self_training()