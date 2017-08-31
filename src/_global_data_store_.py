# These are for globally storing dictionaries
# 1. These are For storing initial train data as labelled and unlabelled
# and also the test data

POS_DICT = {}
NEG_DICT = {}
NEU_DICT = {}
UNLABELED_DICT = {}
TEST_DICT = {}

# 2. These are for storing generated vectors and labels corresponding
# to train data and MODEL, SCALAR, NORMALIZER are related with final model
# for SVM classifier

VECTORS = []
LABELS = []

MODEL = None
SCALAR = None
NORMALIZER = None

CURRENT_ITERATION = 1
LEN_POS = 0
LEN_NEG = 0
LEN_NEU = 0
LEN_TEST = 0


# These are for temporary storing things for SELF TRAINING
# 1. These are for storing LABEL train data in every iteration
# no continuous accumulation will be happen at here.

POS_DICT_SELF = {}
NEG_DICT_SELF = {}
NEU_DICT_SELF = {}

# 2. These are for storing VECTORS and LABELS for the lines generated in
# self_training

VECTORS_SELF = []
LABELS_SELF = []


# These are unigram storing dictionaries for selftraining and normal

POS_UNI_GRAM = {}
NEG_UNI_GRAM = {}
NEU_UNI_GRAM = {}

POS_POST_UNI_GRAM = {}
NEG_POST_UNI_GRAM = {}
NEU_POST_UNI_GRAM = {}

POS_UNI_GRAM_SELF = {}
NEG_UNI_GRAM_SELF = {}
NEU_UNI_GRAM_SELF = {}

POS_POST_UNI_GRAM_SELF = {}
NEG_POST_UNI_GRAM_SELF = {}
NEU_POST_UNI_GRAM_SELF = {}