from pathlib import Path

MAIN_DIR = Path(__file__).parent
PACKAGE_DIR = MAIN_DIR.parent
DATA_DIR = PACKAGE_DIR / 'data'
MODEL_DIR = PACKAGE_DIR / 'model'
OUTPUT_DIR = PACKAGE_DIR / 'output'
LOGS_DIR = OUTPUT_DIR / 'logs'

CMU_DICT_FILE = DATA_DIR / 'cmudict-0.7b'

CMU_CSV = DATA_DIR / 'cmudict-0.7b.csv'

TRAIN_DATASET = DATA_DIR / 'train_dataset.cloudpickle'
VALID_DATASET = DATA_DIR / 'valid_dataset.cloudpickle'
TEST_DATASET = DATA_DIR / 'test_dataset.cloudpickle'
TINY_DATASET = DATA_DIR / 'tiny_dataset.cloudpickle'

LETTERS_YAML = DATA_DIR / 'letters.yaml'
PHONEMES_YAML = DATA_DIR / 'phonemes.yaml'

LETTERS_VOCAB_DICT = DATA_DIR / 'letters_vocab_dict.yaml'
PHONEMES_VOCAB_DICT = DATA_DIR / 'phonemes_vocab_dict.yaml'


if __name__ == '__main__':
    print(MAIN_DIR)
    print(PACKAGE_DIR)
    print()
