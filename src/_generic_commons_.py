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


def get_divided_value(numerator, denominator):
    if denominator == 0:
        return 0.0
    else:
        result = numerator/(denominator * 1.0)
        return round(result, 4)

