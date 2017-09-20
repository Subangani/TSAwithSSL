# Twitter Sentimental Analysis with Semi Supervised Learning
### Introduction
Sentiment Analysing using Semi supervised Learning(SSL) is currently an emerging research and has been found as very use ful compared to supervised and unsupervised.
There are several Approaches found in SSL, such as wrapper based, topic based and graph based. We are currently focusing on self training which is one of wrapper based
technique.

### Methodology
In Self training we have tried with and without replacement, and with replacement works better compared to with replacement. And we follow choosing unlabeled data based on prediction considering the ratio for semi-supervised learning from generated model using a fixed label data following ratios 1%, 5%, 10 %, 20 % and 40 %. While choosing unlabelled data we followed 3 criteria in addition to generic prediction.

1. Analysing the prediction probability Choosing if the probability is exceeds 50 % or greater for particular class and prediction too matches.
1. Predict the probabilities for all unlabeled data and sort in descending order
1. Adding weights to unlabelled data at initial and if a data being used in iteration but doesn’t make sense in fscore average then reduce it weight a factor.

First is used to avoid data which have nearly equal chance of possibly and predicted as a specific class. Second one is for obtain high precision data at initial iteration to make model to have good enough precision to improve the recall and eventually improves the fscore. Last one is to avoid the possibility of repeating same set of unlabeled data to be added as we are following with replacement technique. 

We later focused on Yarowsky based approaches which perform better compared to the above prescribed methodologies and also we tried dividing whole label set as 3 and used 1/3 for initial label and next 2/3 for unlabel and 1/3 for tuning and obtained best parameters and made the prediction.

### Techniques/Technologies Used
Measures in SVM of Scikit Learn
We are using SVC of Scikit Learn’s SVM. SVC stands for Support Vector Classification, There are several other ways and methodologies present in SVM. We are generally uses SVC and mainly contributed on several attributes of SVC such as C parameter  , Gamma, kernel and probability enabled. And we can make different measures to make predictions. 

Accordingly we can focus 3 facts.
1. Decision Function
1. Prediction Probability
1. Prediction

Find out which accurate over other is bit difficult but we can conclude on some point on predicting this based on three approaches. And also in cases there can’t come to same output in 3 methodology for same model and same test data.(link)

### Results

Accordingly we used Semeval data’s train set as labeled and unlabeled. We first use initial labeled as a percentage of total semeval and tested the remaining data by assuming them as unlabeled and obtained set of tweets to be added in each iteration by maintaining ratio between polarity as same as whole data set.( Thus positive 0.34% negative 15% and neutral 51%). In each iteration we add 20 tweets and if it increase the average f score we add them to train set else add them back to unlabeled data and also reduce the weight for those unlabeled data. As when adding tweets in every iteration we consider prediction probability as primary key for sorting and see weight as secondary key and get sorted list and from that only we get the unlabeled data to be used for every iteration.
We tested with following cases and obtained following results

|Percentage of Semeval data used as initial label data|Initial  F score of the supervised svm|Highest F score obtain out of first 20 iterations|Relative improvement after use of self training|Successful Iterations out of first 20 iterations|
|:-:|:-:|:-:|:-:|:-:|
|0.5|0.5023|0.5054|0.31%|9|
|0.4|0.5047|0.5095|0.38%|9|
|0.2|0.466|0.4749|0.89%|6|
|0.1|0.4327|0.4482|1.55%|13|
|0.01|0.386|0.4732|8.72%|8|

Yarowsky based methods are better perfom compared to the adding unlabel in iteration and keep based on f score improvement, and reduce weight if not and continue iteration, due to label to unlabel ratio we face issues increasing f score in few iteration or increase f score with large amount of unlabeled data. But Yarowsky able to out perform all these, and In addition when using divided ( divided by 3 as 0,3,6 for label; 1,4,7 for unlabel and 2,5,8 for tune ) n datas it perform better compared to obtain initial  n datas.



