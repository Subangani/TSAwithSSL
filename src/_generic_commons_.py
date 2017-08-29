import csv
import _config_globals_ as globals


def temp_difference_cal(time_list):
    """
    This function is used when a set of time values are added and difference between last two are obtained
    :param time_list:
    :return: difference
    """
    if len(time_list) > 1:
        final = float(time_list[len(time_list) - 1])
        initial = float(time_list[len(time_list) - 2])
        difference = final - initial
    else:
        difference = -1.0
    return difference


def dict_update(original, temp):
    """
    This will update original dictionary key, and values by comparing with temp values
    :param original:
    :param temp:
    :return: original updated dictionary and a success statement
    """
    is_success = False
    result = {}
    original_temp = original.copy()
    for key in temp.keys():
        global_key_value = original_temp.get(key)
        local_key_value = temp.get(key)
        if key not in original_temp.keys():
            result.update({key:local_key_value})
        else:
            result.update({key: local_key_value + global_key_value})
            del original_temp[key]
    result.update(original_temp)
    return result,is_success


def get_score(test_dict):
    """
    :param test_dict:
    :return:
    """
    TP = TN = TNeu = FP_N = FP_Neu = FN_P = FN_Neu = FNeu_P = FNeu_N = 0
    if len(test_dict) > 0:
        dic = {'positive': 2.0, 'negative': -2.0, 'neutral': 0.0}
        for key in test_dict.keys():
            line = test_dict.get(key)
            new = str(line[2])
            old = str(dic.get(line[0]))
            if old == new:
                if new == "2.0":
                    TP += 1
                elif new == "-2.0":
                    TN += 1
                elif new == "0.0":
                    TNeu += 1
            else:
                if new == "2.0" and old == "-2.0":
                    FP_N += 1
                elif new == "2.0" and old == "0.0":
                    FP_Neu += 1
                elif new == "-2.0" and old == "2.0":
                    FN_P += 1
                elif new == "-2.0" and old == "0.0":
                    FN_Neu += 1
                elif new == "0.0" and old == "2.0":
                    FNeu_P += 1
                elif new == "0.0" and old == "-2.0":
                    FNeu_N += 1
    else:
        print "No test data"
    accuracy = get_divided_value((TP + TN + TNeu),(TP + TN + TNeu + FP_N + FP_Neu + FN_P +FN_Neu + FNeu_P + FNeu_N))
    pre_p = get_divided_value(TP, (FP_N + FP_Neu + TP))
    pre_n = get_divided_value(TN, (FN_P + FN_Neu + TN))
    pre_neu = get_divided_value(TNeu, (FNeu_P + FNeu_N + TNeu))
    re_p = get_divided_value(TP, (FN_P + FNeu_P + TP))
    re_n = get_divided_value(TN, (FP_N + FNeu_N + TN))
    re_neu = get_divided_value(TNeu, (FNeu_P + FNeu_N + TNeu))
    f_score_p = 2 * get_divided_value((re_p * pre_p),(re_p + pre_p))
    f_score_n = 2 * get_divided_value((re_n * pre_n),(re_n + pre_n))
    f_score = round((f_score_p + f_score_n)/2,4)
    return accuracy, pre_p, pre_n, pre_neu, re_p, re_n, re_neu, f_score_p, f_score_n,f_score


def get_divided_value(numerator, denominator):
    if denominator == 0:
        return 0.0
    else:
        result = numerator/(denominator * 1.0)
        return round(result, 4)


def generate_dictionaries(type):
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
    return pos_dict,neg_dict,neu_dict,unlabel_dict

