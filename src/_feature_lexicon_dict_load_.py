import _feature_lexicon_load_commons_  as load_commons

POS_LEXICON_FILE = "../resource/positive.txt"
NEG_LEXICON_FILE = "../resource/negative.txt"
POS_WORDS = load_commons.load_word_dictionary(POS_LEXICON_FILE)
NEG_WORDS = load_commons.load_word_dictionary(NEG_LEXICON_FILE)

AFFIN_FILE_96 = "../resource/afinn_96.txt"
AFFIN_LOAD_96 = load_commons.load_afinn_dictionary(AFFIN_FILE_96)

AFFIN_FILE_111 = "../resource/afinn_111.txt"
AFFIN_LOAD_111 = load_commons.load_afinn_dictionary(AFFIN_FILE_111)

SENTI_140_UNIGRAM_FILE = "../resource/senti140/unigram.txt"
SENTI_140_UNIGRAM_DICT = load_commons.load_generic_dictionary(SENTI_140_UNIGRAM_FILE)

NRC_UNIGRAM_FILE = "../resource/nrc/unigram.txt"
NRC_HASHTAG_FILE = "../resource/nrc/sentimenthash.txt"
NRC_UNIGRAM_DICT = load_commons.load_generic_dictionary(NRC_UNIGRAM_FILE)
NRC_HASHTAG_DICT = load_commons.load_generic_dictionary(NRC_HASHTAG_FILE)

BING_LIU_FILE = "../resource/BingLiu.csv"
BING_LIU_DICT = load_commons.load_generic_dictionary(BING_LIU_FILE)

SENTI_WORD_NET_FILE = "../resource/sentiword_net.txt"
SENTI_WORD_NET_DICT = load_commons.load_senti_word_net_dictionary(SENTI_WORD_NET_FILE)
