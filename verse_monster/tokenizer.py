import copy
from pathlib import Path
from typing import *

from contextlib2 import contextmanager
from transformers import PreTrainedTokenizer, BatchEncoding, TensorType
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import EncodedInput

from verse_monster import utils, constants


class CharPhonemeTokenizer(PreTrainedTokenizer):
    BOS_ID = 0
    PAD_ID = 1
    SEP_ID = 2
    UNK_ID = 3

    _bos_token = '<s>'
    _pad_token = '<pad>'
    _sep_token = '</s>'
    _unk_token = '<unk>'

    def __init__(
            self,
            **kwargs
    ):
        kwargs['model_input_names'] = \
            ['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels']

        super().__init__(**kwargs)

        self.langs = ['en-char', 'en-phoneme']

        # taken from FSMT
        specials = {
            self._bos_token: self.BOS_ID,
            self._pad_token: self.PAD_ID,
            self._sep_token: self.SEP_ID,
            self._unk_token: self.UNK_ID,
        }

        def _yaml_file_to_vocab_dict(filename, offset=len(specials)):
            tokens = utils.load_yaml(filename)

            token_to_id = {
                c: i + offset for i, c in enumerate(tokens)
            }

            out = copy.copy(specials)
            out.update(token_to_id)
            return out

        self.src_tok_2_id = _yaml_file_to_vocab_dict(constants.LETTERS_YAML)
        self.src_id_2_tok = {v: k for k, v in self.src_tok_2_id.items()}

        self.tgt_tok_2_id = _yaml_file_to_vocab_dict(constants.PHONEMES_YAML)
        self.tgt_id_2_tok = {v: k for k, v in self.tgt_tok_2_id.items()}

        self.use_src_lang: Optional[bool] = True

    # def pad(
    #     self,
    #     encoded_inputs: Union[
    #         BatchEncoding,
    #         List[BatchEncoding],
    #         Dict[str, EncodedInput],
    #         Dict[str, List[EncodedInput]],
    #         List[Dict[str, EncodedInput]],
    #     ],
    #     padding: Union[bool, str, PaddingStrategy] = True,
    #     max_length: Optional[int] = None,
    #     pad_to_multiple_of: Optional[int] = None,
    #     return_attention_mask: Optional[bool] = None,
    #     return_tensors: Optional[Union[str, TensorType]] = None,
    #     verbose: bool = True,
    # ) -> BatchEncoding:
    #     pass

    @property
    def vocab_size(self) -> int:
        return self.src_vocab_size

    @property
    def src_vocab_size(self) -> int:
        return len(self.src_tok_2_id)

    @property
    def tgt_vocab_size(self) -> int:
        return len(self.tgt_tok_2_id)

    # def prepare_for_tokenization(
    #     self, text: str, is_split_into_words: bool = False, **kwargs
    # ) -> Tuple[str, Dict[str, Any]]:
    #     # self.use_src_lang: bool = kwargs.pop('use_src_lang')
    #     return text, kwargs

    def _tokenize(self, text: str, **kwargs):
        # print(f'_tokenize: self.use_src_lang: {self.use_src_lang}')
        if self.use_src_lang:
            text = [c for c in text]
        else:
            text = text.split()

        return text + [self.sep_token]

    def _convert_token_to_id(self, token):
        if self.use_src_lang:
            converter_dict = self.src_tok_2_id
        else:
            converter_dict = self.tgt_tok_2_id
        return converter_dict.get(token, self.UNK_ID)

    def _convert_id_to_token(self, index: int) -> str:
        return self.tgt_id_2_tok.get(index, self._unk_token)

    def get_vocab(self) -> Dict[str, int]:
        return self.src_tok_2_id

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, str]:
        src_filename = str(Path(save_directory) / f'{filename_prefix}_src.yaml')
        tgt_filename = str(Path(save_directory) / f'{filename_prefix}_tgt.yaml')

        utils.save_yaml(self.src_tok_2_id, src_filename)
        utils.save_yaml(self.tgt_tok_2_id, tgt_filename)

        return src_filename, tgt_filename

    @contextmanager
    def as_target_tokenizer(self):
        self.use_src_lang = False
        yield
        self.use_src_lang = True


if __name__ == '__main__':
    tok = CharPhonemeTokenizer()

    src = 'abate'
    tgt = 'ah0 b ey1 t'
    print(len(tgt))

    print(tok(src, use_src_lang=True))
    print(tok(tgt, use_src_lang=False))

    # print(tok.build_inputs_with_special_tokens(src, 'ah0 b ey1 t'))
    out = tok.prepare_seq2seq_batch([src], [tgt], use_src_lang=True)
    print(out)
    print(len(out['labels'][0]))



