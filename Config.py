import transformers
import torch

MAX_LEN = 100
EPOCH = 10
MODEL_PATH = "../Save Model/"
LOAD_MODEL = True
TRAINING_FILE = "../Data/train.csv"
TESTING_FILE = "../Data/test.csv/"
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 12
NUMBER_OF_LABEL = 6
MODEL_BASE_PATH = "albert-large-v2"
CLASSIFIER_DROPOUT_PROB = 0.3
HIDDEN_SIZE = 1024
TOKENIZER = transformers.AlbertTokenizer.from_pretrained(MODEL_BASE_PATH,do_lower_case = True)
DEVICE = "cuda" if torch.cuda.is_available() else "cup "
