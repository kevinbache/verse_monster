from typing import *
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel, BatchEncoding
from transformers.file_utils import PaddingStrategy

from verse_monster import tokenizer


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
    label_pad_token_id: int = 1

    @classmethod
    def _pad(
            cls,
            datapoints: List[torch.Tensor],
            padding_side: str = 'right',
            pad_token_id: int = 1,
    ):
        max_label_length = max(len(l) for l in datapoints)
        for i, datapoint in enumerate(datapoints):
            remainder = torch.tensor([pad_token_id] * (max_label_length - len(datapoint)), dtype=torch.long)
            if padding_side == 'right':
                order = [datapoint, remainder]
            else:
                order = [remainder, datapoint]
            datapoints[i] = torch.cat(order).unsqueeze(0)
        out = torch.cat(datapoints, dim=0)
        return out

    def __call__(self, data: List[Dict[str, torch.Tensor]]):
        data = {key: [dp[key] for dp in data] for key in data[0].keys()}

        for k in data:
            data[k] = self._pad(data[k], padding_side='right', pad_token_id=1)

        return data

    def convert_tokens_to_ids(self, tokens):
        self.tokenizer.convert_tokens_to_ids(tokens)
