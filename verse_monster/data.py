import pandas as pd

from verse_monster import constants, utils


def cmu_dict_2_csv():
    DROP_IF_STARTS_WITH = [';;;', '"""', '!', '"', '(', ')', '#', '%', '&', '+', '-', '.', '/', ':', ';', '?', '{', '}']
    DROP_IF_CONTAINS = ['_', '�', ',']

    rows = []
    all_letters = set()
    all_phonemes = set()

    with open(constants.CMU_DICT_FILE, 'r') as f:
        for li, line in enumerate(f):
            skip_line = False
            for c in DROP_IF_STARTS_WITH:
                if line.startswith(c):
                    skip_line = True
                    break
            if skip_line:
                continue

            for c in DROP_IF_CONTAINS:
                if c in line:
                    skip_line = True
                    break
            if skip_line:
                continue

            # remove trailing \n
            line = line.strip()

            tokens = [t.lower() for t in line.split(' ') if t]
            letters = tokens[0]
            tokens = tokens[1:]

            if letters.endswith(')'):
                letters = letters[:-3]

            rows.append({
                'letters': letters,
                'phonemes': ' '.join(tokens),
            })
            all_letters.update(letters)
            all_phonemes.update(tokens)

    df = pd.DataFrame(rows)
    df.to_csv(constants.CMU_CSV, index=False)

    all_letters = sorted(list(all_letters))
    all_phonemes = sorted(list(all_phonemes))

    utils.save_yaml(all_letters, constants.LETTERS_YAML)
    utils.save_yaml(all_phonemes, constants.PHONEMES_YAML)

    print(len(all_letters), all_letters)
    print(len(all_phonemes), all_phonemes)


if __name__ == '__main__':
    cmu_dict_2_csv()
