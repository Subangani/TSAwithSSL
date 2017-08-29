import _feature_lexicon_load_commons_  as load_commons

POS_LEXICON_FILE = "../resource/positive.txt"
NEG_LEXICON_FILE = "../resource/negative.txt"
POS_WORDS = load_commons.load_words(POS_LEXICON_FILE)
NEG_WORDS = load_commons.load_words(NEG_LEXICON_FILE)

AFFIN_FILE_96 = "../resource/afinn_96.txt"
AFFIN_LOAD_96 = load_commons.load_afinn(AFFIN_FILE_96)

AFFIN_FILE_111 = "../resource/afinn_111.txt"
AFFIN_LOAD_111 = load_commons.load_afinn(AFFIN_FILE_111)

SENTI_140_UNIGRAM_FILE = "../resource/senti140/unigrams-pmilexicon.txt"
SENTI_140_UNIGRAM_DICT = load_commons.create_dict(SENTI_140_UNIGRAM_FILE)

NRC_UNIGRAM_FILE = "../resource/nrc/unigrams-pmilexicon.txt"
NRC_HASHTAG_FILE = "../resource/nrc/sentimenthashtags.txt"
NRC_UNIGRAM_DICT = load_commons.create_dict(NRC_UNIGRAM_FILE)
NRC_HASHTAG_DICT = load_commons.create_dict(NRC_HASHTAG_FILE)

BING_LIU_FILE = "../resource/BingLiu.csv"
BING_LIU_DICT = load_commons.create_dict(BING_LIU_FILE)

SENTI_WORD_NET_FILE = "../resource/sentiword_net.txt"
SENTI_WORD_NET_DICT = load_commons.load_senti_word_net(SENTI_WORD_NET_FILE)



