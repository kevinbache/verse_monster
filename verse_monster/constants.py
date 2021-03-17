from pathlib import Path

MAIN_DIR = Path(__file__).parent
PACKAGE_DIR = MAIN_DIR.parent
DATA_DIR = PACKAGE_DIR / 'data'

CMU_DICT_FILE = DATA_DIR / 'cmudict-0.7b'

CMU_CSV = DATA_DIR / 'cmudict-0.7b.csv'

LETTERS_YAML = DATA_DIR / 'letters.yaml'
PHONEMES_YAML = DATA_DIR / 'phonemes.yaml'


if __name__ == '__main__':
    print(MAIN_DIR)
    print(PACKAGE_DIR)
    print()
