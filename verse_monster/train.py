import copy
import logging
from typing import *

import numpy as np
from datasets import load_metric
from torch import nn
from transformers import (
    FSMTForConditionalGeneration,
    FSMTConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    IntervalStrategy, TrainerCallback
)

from verse_monster import constants, utils, tokenizer
from verse_monster.collator import MySeq2SeqCollator


logger = logging.getLogger(__name__)
logger.setLevel('INFO')


def copy_weights(from_model: nn.Module, to_model: nn.Module, layer_names_to_skip: List[str], verbose=True):
    from_sd = from_model.state_dict()
    to_sd = to_model.state_dict()
    for k, v in to_sd.items():
        if verbose:
            print(f"{k:<60}: my shape: {str(v.shape):<30}", end='')

        if k in layer_names_to_skip:
            print(f"  enru SKIPPED =========================================================")
        else:
            print(f"  enru shape: {str(from_sd[k].shape):<30}")
            if from_sd[k].shape != to_sd[k].shape:
                print("          DIFFERENT SHAPES")
            to_sd[k] = copy.copy(from_sd[k])


def freeze_weights(model: nn.Module, layers_to_skip: List[str], do_learn_layer_norms=True, do_print_count=True):
    for name, param in model.named_parameters(recurse=True):
        if name in layers_to_skip or (do_learn_layer_norms and 'layer_norm' in name):
            continue
        param.requires_grad = False

    if do_print_count:
        count = sum(param.numel() for param in model.parameters(recurse=True) if param.requires_grad)
        print(f'Number of trainable parameters: {count}')


def unfreeze_weights(model: nn.Module, layers_to_skip: Optional[List[str]] = None):
    # if layers_to_skip is None:
    #     layers_to_skip = []

    for name, param in model.named_parameters(recurse=True):
        # if name in layers_to_skip:
        #     continue
        param.requires_grad = True


def prep_model(
        my_model: nn.Module,
        model_with_pretrained_weights: nn.Module,
        layer_names_to_learn: List[str],
        do_learn_layer_norms=True,
        verbose=True,
):
    copy_weights(model_with_pretrained_weights, my_model, layer_names_to_skip=layer_names_to_learn, verbose=verbose)
    freeze_weights(my_model, layers_to_skip=layer_names_to_learn, do_learn_layer_norms=do_learn_layer_norms)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


# noinspection DuplicatedCode
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)
    # if data_args.ignore_pad_token_for_loss:
    #     # Replace -100 in the labels as we can't decode them.
    #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tok.PAD_ID) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def convert_dataset_to_lists(dataset):
    for i, dp in enumerate(dataset):
        for k in dp:
            dataset[i][k] = dataset[i][k].tolist()
    return dataset


def remove_keys(dataset, keys_to_remove):
    dp = {}
    for dp in dataset:
        for k in keys_to_remove:
            if k in dp:
                # print(f'removing {k}')
                del dp[k]
    print(f'dp.keys(): {dp.keys()}')


def limit_datset(ds, num_datapoints):
    ds.data = ds.data[:num_datapoints]
    ds.meta = ds.meta[:num_datapoints]
    return ds


WEIGHTS_MODEL_NAME = 'facebook/wmt19-en-ru'


if __name__ == '__main__':
    do_recreate_my_model = False
    do_test_run = False

    batch_size = 8

    num_beams = 4

    num_train = 1000
    num_valid = 100

    if do_test_run:
        with utils.Timer('loading tiny datset'):
            ds_tiny = utils.load_cloudpickle(constants.TINY_DATASET)
            ds_train = ds_tiny
            ds_valid = ds_tiny
    else:
        with utils.Timer('loading datasets'):
            ds_train = utils.load_cloudpickle(constants.TRAIN_DATASET)
            ds_valid = utils.load_cloudpickle(constants.VALID_DATASET)
            # ds_test = utils.load_cloudpickle(constants.TEST_DATASET)

            if num_train is not None:
                ds_train = limit_datset(ds_train, num_train)
            if num_valid is not None:
                ds_valid = limit_datset(ds_train, num_valid)

            keys_to_remove = ('decoder_attention_mask', 'decoder_input_ids')
            # keys_to_remove = ('decoder_attention_mask', )
            # keys_to_remove = ()
            remove_keys(ds_train, keys_to_remove)
            remove_keys(ds_valid, keys_to_remove)

            """
            attn_mask
            tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
                    [0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
                    [0., 0., 0., -inf, -inf, -inf, -inf, -inf],
                    [0., 0., 0., 0., -inf, -inf, -inf, -inf],
                    [0., 0., 0., 0., 0., -inf, -inf, -inf],
                    [0., 0., 0., 0., 0., 0., -inf, -inf],
                    [0., 0., 0., 0., 0., 0., 0., -inf],
                    [0., 0., 0., 0., 0., 0., 0., 0.]])
            torch.Size([8, 8])
            attn_weights
            # (batch x nHeads x nDecTokens x nDecTokens)
            torch.Size([7, 16, 8, 8])
            attn_weights 2
            torch.Size([112, 8, 8])
            """

            # ds_tiny = ds_valid[:10]
            # utils.save_cloudpickle(ds_tiny, constants.TINY_DATASET)
            # raise ValueError('')

    with utils.Timer('loading CharPhonemeTokenizer'):
        tok = tokenizer.CharPhonemeTokenizer()

    layer_names_to_learn = [
        'model.encoder.embed_tokens.weight',
        'model.encoder.embed_positions.weight',
        'model.decoder.output_projection.weight',
        'model.decoder.embed_tokens.weight',
        'model.decoder.embed_positions.weight',
    ]

    if do_recreate_my_model:
        with utils.Timer('loading_enru model'):
            enru_model = FSMTForConditionalGeneration.from_pretrained(WEIGHTS_MODEL_NAME)

        with utils.Timer('creating my_model from config'):
            my_model = FSMTForConditionalGeneration(FSMTConfig(
                langs=['en-char', 'en-phoneme'],
                src_vocab_size=tok.src_vocab_size,
                tgt_vocab_size=tok.tgt_vocab_size,
                max_position_embeddings=64,  # empirical maxes are 34 for chars and 32 for phonemes
                # match enru
                encoder_layers=6,
                decoder_layers=6,
                encoder_ffn_dim=1024 * 8,
            ))

        prep_model(my_model, enru_model, layer_names_to_learn, do_learn_layer_norms=True)

        with utils.Timer('saving model'):
            my_model.save_pretrained(save_directory=constants.MODEL_DIR)
    else:
        with utils.Timer('loading model'):
            my_model = FSMTForConditionalGeneration.from_pretrained(constants.MODEL_DIR, local_files_only=True)
            freeze_weights(my_model, layers_to_skip=layer_names_to_learn, do_learn_layer_norms=True)

    data_collator = MySeq2SeqCollator(
        tokenizer=tok,
        model=my_model,
        padding='longest',
        label_pad_token_id=tok.PAD_ID,
    )

    # Metric
    metric = load_metric("sacrebleu")

    from transformers import SchedulerType, TrainingArguments
    trainer_args = Seq2SeqTrainingArguments(
        output_dir=constants.OUTPUT_DIR,          # output directory
        logging_dir=constants.LOGS_DIR,           # directory for storing logs
        num_train_epochs=4,                       # total # of training epochs
        # max_steps=100,
        per_device_train_batch_size=batch_size,   # batch size per device during training
        per_device_eval_batch_size=batch_size,    # batch size for evaluation
        warmup_steps=500,                         # number of warmup steps for learning rate scheduler
        learning_rate=1e-3,
        weight_decay=0.01,                        # strength of weight decay
        predict_with_generate=True,
        sortish_sampler=True,
        do_eval=True,
        do_predict=True,
        evaluation_strategy=IntervalStrategy.EPOCH,
        # eval_steps=50,
        dataloader_num_workers=4,
        report_to=['none'],
        lr_scheduler_type=SchedulerType.CONSTANT_WITH_WARMUP,
        logging_first_step=True,
    )

    trainer = Seq2SeqTrainer(
        model=my_model,
        args=trainer_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_out = trainer.train()
    print(train_out)

    eval_out = trainer.evaluate(
        eval_dataset=ds_valid,
        max_length=40,
        num_beams=num_beams,
    )
    print(eval_out)

    num_preds = 10

    predict_out = trainer.predict(
        test_dataset=ds_valid[:num_preds],
        max_length=40,
        num_beams=num_beams,
        ignore_keys=['decoder_input_ids,', ]
    )
    print(predict_out)

    with tok.as_target_tokenizer():
        preds = tok.batch_decode(predict_out.predictions)

    print('Predictions:')
    for meta, pred_str, pred_tokens in zip(ds_valid.meta, preds, predict_out.predictions):
        print(meta['letters'], pred_str[:30], pred_tokens[:10])
