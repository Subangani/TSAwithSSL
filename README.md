# Twitter Sentimental Analysis with Semi Supervised Learning
### Introduction
Sentiment Analysing using Semi supervised Learning(SSL) is currently an emerging research and has been found as very use ful compared to supervised and unsupervised.
There are several Approaches found in SSL, such as wrapper based, topic based and graph based. We are currently focusing on self training which is one of wrapper based
technique.

### Methodology
We are currently following Yarowsky(1995) algorithm
and obtain results but that doesn't make significant impact 
over large data and we are currently having larger labeled which should be vise versa
in real cases.

In Self training we have tried with and without replacement, and with replacement works better compared to with replacement. And we follow choosing unlabeled data based on prediction considering the ratio for semi-supervised learning from generated model using a fixed label data following ratios 1%, 5%, 10 %, 20 % and 40 %. While choosing unlabelled data we followed 3 criteria in addition to generic prediction.

1. Analysing the prediction probability Choosing if the probability is exceeds 50 % or greater for particular class and prediction too matches.
1. Predict the probabilities for all unlabeled data and sort in descending order
1. Adding weights to unlabelled data at initial and if a data being used in iteration but doesn’t make sense in fscore average then reduce it weight a factor.

First is used to avoid data which have nearly equal chance of possibly and predicted as a specific class. Second one is for obtain high precision data at initial iteration to make model to have good enough precision to improve the recall and eventually improves the fscore. Last one is to avoid the possibility of repeating same set of unlabeled data to be added as we are following with replacement technique. 

### Results
[Details will be Updated later]


