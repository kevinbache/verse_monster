import copy
from collections import defaultdict
from typing import List, Dict, Any

import pandas as pd
import torch
from torch.utils.data import IterableDataset, Dataset

from verse_monster import constants, utils, tokenizer


DROP_IF_STARTS_WITH = [';;;', '"""', '!', '"', '(', ')', '#', '%', '&', '+', '-', '.', '/', ':', ';', '?', '{', '}']
DROP_IF_CONTAINS = ['_', '�', ',']


def cmu_dict_2_csv():
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


class ListMetaDataset(Dataset):
    def __init__(self, list_of_datapoints: List[Dict[str, Any]], list_of_meta: List[Dict[str, Any]]):
        super().__init__()
        assert (len(list_of_datapoints) == len(list_of_meta))
        self.data = list_of_datapoints
        self.meta = list_of_meta

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def copy(self, num_datapoints: int):
        return self.__class__(
            list_of_datapoints=copy.deepcopy(self.data[:num_datapoints]),
            list_of_meta=copy.deepcopy(self.meta[:num_datapoints]),
        )


if __name__ == '__main__':
    seed = 1234

    cmu_dict_2_csv()

    tok = tokenizer.CharPhonemeTokenizer()

    # tok.prepare_seq2seq_batch(src_texts=['abate'], tgt_texts=['aa0 aa1 aa2'])

    with utils.Timer('reading, applying'):
        df = pd.read_csv(constants.CMU_CSV)
        df['letters_tok'] = \
            df['letters'].apply(lambda x: tok(str(x), return_token_type_ids=False, return_tensors='pt'))

        with tok.as_target_tokenizer():
            df['phonemes_tok'] = \
                df['phonemes'].apply(lambda x: tok(str(x), return_token_type_ids=False, return_tensors='pt'))

        print('letters_tok: ', df['letters_tok'].iloc[0])
        print('phonemes_tok: ', df['phonemes_tok'].iloc[0])
        print()

    with utils.Timer('making datapoints and metas'):
        datapoints = []
        metas = []
        for ind, row in df.iterrows():
            dp = row['letters_tok']
            dp[constants.DataNames.LABELS] = row['phonemes_tok'][constants.DataNames.INPUT_IDS]
            for k in dp:
                dp[k] = dp[k].squeeze()

            datapoints.append(dp)
            metas.append({
                'index': ind,
                'letters': row['letters'],
                'phonemes': row['phonemes'],
            })

        print(f'len(datapoints): {len(datapoints)}')
        print(f'len(metas): {len(metas)}')
        print(datapoints[0])
        print(metas[0])

    # pd.set_option('display.max_columns', 10)
    # pd.set_option('display.width', 200)
    # print(df.head(10))

    # group by letters so we don't have different pronunciations for the same word in train vs valid/test
    with utils.Timer('grouping...'):
        letters_to_rows = defaultdict(list)
        for meta, dp in zip(metas, datapoints):
            letters_to_rows[meta['letters']].append((dp, meta))
        print(f'len letterstorows: {len(letters_to_rows)}')
        del meta, datapoints

    # len(df)
    # Out[8]: 134222
    # len(letters_to_rows)
    # Out[7]: 125631

    with utils.Timer('making datapoints_and_meta...'):
        datapoints_and_meta = list(letters_to_rows.values())
        print(len(datapoints_and_meta))
        print(datapoints_and_meta[0])
        del letters_to_rows
        print('splitting...')
        train, valid, test = utils.train_valid_test_split(datapoints_and_meta, seed=seed)
        del datapoints_and_meta
        print('pre_flatten lens: ', len(train), len(valid), len(test))

    with utils.Timer('flattening...'):
        train = utils.flatten_list_of_lists(train)
        valid = utils.flatten_list_of_lists(valid)
        test = utils.flatten_list_of_lists(test)

    with utils.Timer('inverting...'):
        train, train_meta = zip(*train)
        valid, valid_meta = zip(*valid)
        test, test_meta = zip(*test)
        print('flattend/inverted lens: ', len(train), len(valid), len(test))

    ds_train = ListMetaDataset(train, train_meta)
    del train, train_meta
    ds_valid = ListMetaDataset(valid, valid_meta)
    del valid, valid_meta
    ds_test = ListMetaDataset(test, test_meta)
    del test, test_meta

    ds_tiny = ds_train.copy(10)

    with utils.Timer('saving...'):
        utils.save_cloudpickle(ds_train, constants.TRAIN_DATASET)
        utils.save_cloudpickle(ds_valid, constants.VALID_DATASET)
        utils.save_cloudpickle(ds_test, constants.TEST_DATASET)
        utils.save_cloudpickle(ds_tiny, constants.TINY_DATASET)

    print('done')
