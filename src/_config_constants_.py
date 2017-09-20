# CONFIGURATION CONSTANTS

CLASSIFIER_SVM = "svm"
KERNEL_LINEAR = "linear"
KERNEL_RBF = "rbf"

DATA_SET_SIZE = 20633
LABEL_POSITIVE = 2.0
LABEL_NEGATIVE = -2.0
LABEL_NEUTRAL = 0.0

POS_RATIO = 0.34213
NEG_RATIO = 0.15659
NEU_RATIO = 0.50128

# HEADER OF THE SAVING FILE

CSV_HEADER = ["PositiveTrainSet", "NegativeTrainSet", "NeutralTrainSet",
              "FeatureSetCode", "TestSetLimit", "Iteration", "Accuracy",
              "PrecisionPositive", "PrecisionNegative", "PrecisionNeutral",
              "RecallPositive", "RecallNegative", "RecallNeutral",
              "fScorePositive", "fScoreNegative", "fScoreAverage"]