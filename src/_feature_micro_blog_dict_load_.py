import _feature_micro_blog_load_commons_ as load_common

EMOTICON_FILE = "../resource/emoticon.txt"
EMOTICON_DICT = load_common.load_emoticon_dictionary(EMOTICON_FILE)

UNICODE_EMOTICON_FILE = "../resource/emoticon.csv"
UNICODE_EMOTICON_DICT = load_common.load_unicode_emoticon_dictionary(UNICODE_EMOTICON_FILE)
