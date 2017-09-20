from __future__ import division

import csv


def load_emoticon_dictionary(filename):
    emo_scores = {'Positive': 0.5, 'Extremely-Positive': 1.0, 'Negative': -0.5, 'Extremely-Negative': -1.0,
                  'Neutral': 0.0}
    emo_score_list = {}
    fi = open(filename, "r")
    l = fi.readline()
    while l:
        l = l.replace("\xc2\xa0", " ")
        li = l.split(" ")
        l2 = li[:-1]
        l2.append(li[len(li) - 1].split("\t")[0])
        sentiment = li[len(li) - 1].split("\t")[1][:-2]
        score = emo_scores[sentiment]
        l2.append(score)
        for i in range(0, len(l2) - 1):
            emo_score_list[l2[i]] = l2[len(l2) - 1]
        l = fi.readline()
    return emo_score_list


def load_unicode_emoticon_dictionary(filename):
    emo_score_list = {}
    with open(filename, "r") as unlabeled_file:
        reader = csv.reader(unlabeled_file)
        for line in reader:
            emo_score_list.update({str(line[0]): float(line[4])})
    return emo_score_list
