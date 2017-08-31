import csv
import os
import time
import _generic_commons_ as commons
import _config_globals_ as globals
import _load_model_test_iterate_ as lmti


def self_training():
    directory = "../dataset/analysed/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    if os.path.exists(directory):
        with open('../dataset/analysed/' + lmti.get_file_prefix() + 'result.csv', 'w') as result:

            csv_result = csv.writer(result)
            csv_result.writerow(globals.CSV_HEADER)

            time_list = [time.time()]
            result = lmti.initial_run()
            print result
            csv_result.writerow(result)
            fscore_best = result[15]

            while lmti.ds.CURRENT_ITERATION <= globals.NO_OF_ITERATION:
                if lmti.ds.CURRENT_ITERATION == 1:
                    is_self_training = False
                else:
                    is_self_training = True

                result = lmti.self_training_run(is_self_training)
                print result
                fscore_iter = result[15]

                if fscore_best <= fscore_iter:
                    lmti.upgrade()
                    fscore_best = fscore_iter
                    result = lmti.get_result(lmti.ds.TEST_DICT)
                    csv_result.writerow(result)
                else:
                    lmti.downgrade()
                lmti.ds.CURRENT_ITERATION += 1
            time_list.append(time.time())
            print commons.temp_difference_cal(time_list)
    else:
        print "Error in access directory ", directory
    return





self_training()