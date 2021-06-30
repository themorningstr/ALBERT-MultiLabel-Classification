import transformers
import torch

MAX_LEN = 120
EPOCH = 3
MODEL_PATH = "../Save Model/model.bin"
TRAINING_FILE = "../Data/train.csv"
TESTING_FILE = "../Data/test.csv"
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 12
NUMBER_OF_LABEL = 6
MODEL_BASE_PATH = "bert-base-multilingual-uncased"
CLASSIFIER_DROPOUT_PROB = 0.3
HIDDEN_SIZE = 1024
TOKENIZER = transformers.BertTokenizer.from_pretrained(MODEL_BASE_PATH,do_lower_case = True)
DEVICE = "cuda" if torch.cuda.is_available() else "cup "
THRESHOLD = 0.5
