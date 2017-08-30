import csv
import os
import time
import _generic_commons_ as commons
import _config_globals_ as globals
import _config_constants_ as cons
import _load_model_test_iterate_ as lmti


def self_training():
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

            lmti.load_initial_dictionaries(1)
            lmti.get_vectors_and_labels()
            lmti.generate_model(is_self_training=False)
            lmti.store_test(is_self_training=False)

            result = lmti.get_result(lmti.ds.TEST_DICT)
            print result
            csv_result.writerow(result)

            fscore_best = result[15]
            while lmti.ds.CURRENT_ITERATION <= no_of_iteration:
                if lmti.ds.CURRENT_ITERATION == 1:
                    is_self_training = False
                else:
                    is_self_training = True

                lmti.load_iteration_dict(is_self_training)
                lmti.get_vectors_and_labels_self()
                lmti.generate_model(is_self_training=True)
                lmti.store_test(is_self_training=True)

                result = lmti.get_result(lmti.ds.TEST_DICT)
                fscore_iter = result[15]

                if fscore_best <= fscore_iter:
                    lmti.ds.POS_UNI_GRAM = lmti.ds.POS_UNI_GRAM_SELF
                    lmti.ds.NEG_UNI_GRAM = lmti.ds.NEG_UNI_GRAM_SELF
                    lmti.ds.NEU_UNI_GRAM = lmti.ds.NEU_UNI_GRAM_SELF
                    lmti.ds.POS_POST_UNI_GRAM = lmti.ds.POS_POST_UNI_GRAM_SELF
                    lmti.ds.NEG_POST_UNI_GRAM = lmti.ds.NEG_POST_UNI_GRAM_SELF
                    lmti.ds.NEU_POST_UNI_GRAM = lmti.ds.NEU_POST_UNI_GRAM_SELF
                    lmti.ds.POS_DICT.update(lmti.ds.POS_DICT_SELF.copy())
                    lmti.ds.NEG_DICT.update(lmti.ds.NEG_DICT_SELF.copy())
                    lmti.ds.NEU_DICT.update(lmti.ds.NEU_DICT_SELF.copy())
                    lmti.ds.VECTORS = lmti.ds.VECTORS_SELF
                    lmti.ds.LABELS = lmti.ds.LABELS_SELF
                    fscore_best = fscore_iter
                    result = lmti.get_result(lmti.ds.TEST_DICT)
                    print result
                    csv_result.writerow(result)
                else:
                    for key in lmti.ds.POS_DICT_SELF.keys():
                        lmti.ds.UNLABELED_DICT.update({str(cons.DATA_SET_SIZE * (lmti.ds.CURRENT_ITERATION + 1) + int(key)): lmti.ds.POS_DICT_SELF.get(key)})
                    for key in lmti.ds.NEG_DICT_SELF.keys():
                        lmti.ds.UNLABELED_DICT.update({str(cons.DATA_SET_SIZE * (lmti.ds.CURRENT_ITERATION + 1) + int(key)): lmti.ds.NEG_DICT_SELF.get(key)})
                    for key in lmti.ds.NEU_DICT_SELF.keys():
                        lmti.ds.UNLABELED_DICT.update({str(cons.DATA_SET_SIZE * (lmti.ds.CURRENT_ITERATION + 1) + int(key)): lmti.ds.NEU_DICT_SELF.get(key)})
                lmti.ds.CURRENT_ITERATION += 1
            time_list.append(time.time())
            print commons.temp_difference_cal(time_list)
    else:
        print "Error in access directory ", directory
    return


self_training()