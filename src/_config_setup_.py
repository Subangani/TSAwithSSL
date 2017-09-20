import _config_constants_ as cons

# This is SETUP interface thus we can adjust input parameters
# Parameters related to training model (some constants are important to
# look nice)

LABEL_RATIO = 1.0/6
TEST_LIMIT = 13000
NO_OF_ITERATION = 10

FEATURE_SET_CODE = 15
# FEATURE_SET is combination of set of features like lexicon, writing style and ngrams
# (also emoticons) according I have defined a set of feature set code and get combinations
# out of these code 15 performs better which include all features.
# To customize you can edit map_tweet(tweet,is_self_training) in _load_model_test_iterate_.py


DEFAULT_CLASSIFIER = cons.CLASSIFIER_SVM

if DEFAULT_CLASSIFIER == cons.CLASSIFIER_SVM:
    DEFAULT_KERNEL = cons.KERNEL_RBF
    DEFAULT_C_PARAMETER = 0.91
    DEFAULT_GAMMA_SVM = 0.03
    DEFAULT_CLASS_WEIGHTS = {2.0: 1.47, 0.0: 1, -2.0: 3.125}
