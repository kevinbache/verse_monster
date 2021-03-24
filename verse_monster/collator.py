from typing import *
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel, BatchEncoding
from transformers.file_utils import PaddingStrategy
from transformers.models.fsmt import modeling_fsmt

from verse_monster import constants
from verse_monster import tokenizer


def make_padded_causal_mask(seq_len: int, padded_len: int):
    """

    Args:
        seq_len: length of unpadded seq
        padded_len: length of padded seq

    Examples:
        >>> make_padded_causal_mask(3, 5)
        tensor([[1., -inf, -inf, -inf, -inf],
            [1., 1., -inf, -inf, -inf],
            [1., 1., 1., -inf, -inf],
            [-inf, -inf, -inf, -inf, -inf],
            [-inf, -inf, -inf, -inf, -inf]])

    Returns: torch.Tensor of size padded_len x padded_len
        -inf everywhere except for a lower triangular matrix of size seq_len in the top left
    """
    out = modeling_fsmt.fill_with_neg_inf(torch.zeros(padded_len, padded_len))
    arange = torch.arange(padded_len)
    rep = arange.unsqueeze(0).repeat(padded_len, 1)
    arange[:seq_len] = arange[:seq_len] + 1
    arange[seq_len:padded_len] = 0
    mask = rep < arange.unsqueeze(-1)
    out = out.masked_fill(mask, 1)
    return out


@dataclass
class MySeq2SeqCollator:
    """
    Based on DataCollatorForSeq2Seq but far simpler because it just targets our particular use case.



    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    @classmethod
    def _pad_1d(cls, tensor_list: List[torch.Tensor], pad_token_id):
        max_length = max(len(l) for l in tensor_list)

        for i, this_tensor in enumerate(tensor_list):
            remainder_len = max_length - len(this_tensor)
            remainder = torch.tensor([pad_token_id] * remainder_len, dtype=torch.long)
            tensor_list[i] = torch.cat([this_tensor, remainder]).unsqueeze(0)

        out = torch.cat(tensor_list, dim=0)
        return out

    def __call__(self, batch: List[Dict[str, torch.Tensor]]):
        # list of dicts --> dict of lists
        batch = {key: [dp[key] for dp in batch] for key in batch[0].keys()}

        # decoder_attention_mask = self.make_decoder_attention_masks(batch)

        for k in batch:
            batch[k] = self._pad_1d(batch[k], pad_token_id=self.label_pad_token_id)

        # batch['decoder_attention_mask'] = decoder_attention_mask

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            labels = batch[constants.DataNames.LABELS]
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=labels)
            batch[constants.DataNames.DECODER_INPUT_IDS] = decoder_input_ids

            # print('collator created decoder input ids:')
            # print(decoder_input_ids)
            # print()

            # print('collator created labels:')
            # print(labels)
            # print()

        return batch

    @staticmethod
    def make_decoder_attention_masks(batch):
        labels = batch[constants.DataNames.LABELS]
        print(f'labels: {labels}')
        print(f'type(labels): {type(labels)}')
        label_lens = [len(l) for l in labels]
        max_len = max(label_lens)
        masks = [make_padded_causal_mask(len(l), max_len).unsqueeze(0) for l in labels]
        decoder_attention_mask = torch.cat(masks, dim=0)
        return decoder_attention_mask

    def convert_tokens_to_ids(self, tokens):
        self.tokenizer.convert_tokens_to_ids(tokens)


if __name__ == '__main__':

    batch = [
        {'input_ids': torch.tensor([39, 21, 25, 18, 28, 21, 2]),
         'attention_mask': torch.tensor([1, 1, 1, 1, 1, 1, 1]),
         'labels': torch.tensor([69, 42, 22, 10, 46, 2])},
        {'input_ids': torch.tensor([17, 34, 35, 31, 30, 25, 35, 36, 35, 2]),
         'attention_mask': torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
         'labels': torch.tensor([5, 57, 58, 10, 48, 10, 58, 60, 58, 2])},
        {'input_ids': torch.tensor([17, 34, 35, 31, 30, 25, 35, 36, 35, 2]),
         'attention_mask': torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
         'labels': torch.tensor([5, 57, 2])}
    ]
    tok = tokenizer.CharPhonemeTokenizer()
    col = MySeq2SeqCollator(
        model=None,
        tokenizer=tok,
        padding=True,
        label_pad_token_id=-100,
    )
    bc = col(batch)
    print(bc)

