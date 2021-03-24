import copy
import logging
from typing import *

import numpy as np
from datasets import load_metric
from torch import nn
from transformers import (
    # FSMTForConditionalGeneration,
    FSMTConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    IntervalStrategy,
)
from torchtext.data.metrics import bleu_score

from verse_monster import constants, utils, tokenizer, fsmt
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
        total_count = sum(param.numel() for param in model.parameters(recurse=True))
        print(f'Number of trainable parameters: {count}')
        print(f'Number of total parameters:     {total_count}')


def unfreeze_weights(model: nn.Module):
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
    # freeze_weights(my_model, layers_to_skip=layer_names_to_learn, do_learn_layer_norms=do_learn_layer_norms)


def postprocess_text(preds, labels):
    preds = [pred.split().strip() for pred in preds if pred]
    labels = [[label.split().strip()] for label in labels if label]

    return preds, labels


# # Metric
metric = load_metric("sacrebleu")


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
    # decoded_preds = [p.split() for p in decoded_preds]
    # decoded_labels = [p.split() for p in decoded_preds]

    # decoded_labels = [[l] for l in decoded_labels]

    print('preds, labels:')
    print(decoded_preds)
    print(decoded_labels)

    # candidate_corpus = [['My', 'full', 'pytorch', 'test']]
    # references_corpus = [[['My', 'full', 'pytorch', 'test']]]
    # bleu_score(candidate_corpus, references_corpus)

    result = {
        "sacrebleu": metric.compute(predictions=decoded_preds, references=decoded_labels)['score'],
    }

    prediction_lens = [np.count_nonzero(pred != constants.INPUTS_PAD_ID) for pred in preds]
    result['bleu'] = bleu_score(candidate_corpus=decoded_preds, references_corpus=decoded_labels)
    result['gen_len'] = np.mean(prediction_lens)

    result = {k: round(v, 3) for k, v in result.items()}
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


def prep_dataset(dataset, keys_to_remove, num_datapoints_to_keep):
    if dataset is None:
        return dataset
    dataset = limit_datset(dataset, num_datapoints_to_keep)
    remove_keys(dataset, keys_to_remove)
    # bos_tensor = torch.tensor([tokenizer.CharPhonemeTokenizer.BOS_ID], dtype=torch.int)
    # for dp in dataset:
    #     # print(f'prep dp: {dp}')
    #     dp['decoder_attention_mask'] = make_causal_mask(len(dp['decoder_attention_mask']))
    #     # dp['labels'] = torch.cat([bos_tensor, dp['labels']])
    # print(f'pos prep dp: {dp}')
    return dataset


if __name__ == '__main__':
    seed = 1234
    np.random.seed(seed)

    do_recreate_my_model = False
    do_test_run = utils.is_local_run()

    batch_size = 8

    num_beams = 4

    num_train = 10000
    num_valid = 400

    ds_train = None
    ds_valid = None
    ds_test = None

    # keys_to_remove = ('decoder_attention_mask', 'decoder_input_ids')
    keys_to_remove = ()

    if do_test_run:
        with utils.Timer('loading tiny datset'):
            ds_tiny = utils.load_cloudpickle(constants.TINY_DATASET)
            ds_train = ds_tiny
            ds_valid = copy.deepcopy(ds_tiny)
            ds_test = copy.deepcopy(ds_tiny)
    else:
        with utils.Timer('loading datasets'):
            # noinspection PyRedeclaration
            ds_train = utils.load_cloudpickle(constants.TRAIN_DATASET)
            # noinspection PyRedeclaration
            ds_valid = utils.load_cloudpickle(constants.VALID_DATASET)
            # noinspection PyRedeclaration
            # ds_test = utils.load_cloudpickle(constants.TEST_DATASET)

            ds_train = prep_dataset(ds_train, keys_to_remove, num_train)
            ds_valid = prep_dataset(ds_valid, keys_to_remove, num_valid)
            ds_test = prep_dataset(ds_test, keys_to_remove, num_valid)

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
            enru_model = fsmt.MyFSMTForConditionalGeneration.from_pretrained(WEIGHTS_MODEL_NAME)

        with utils.Timer('creating my_model from config'):
            my_model = fsmt.MyFSMTForConditionalGeneration(FSMTConfig(
                langs=['en-char', 'en-phoneme'],
                src_vocab_size=tok.src_vocab_size,
                tgt_vocab_size=tok.tgt_vocab_size,
                max_position_embeddings=64,  # empirical maxes are 34 for chars and 32 for phonemes
                # match enru
                encoder_layers=6,
                decoder_layers=6,
                encoder_ffn_dim=1024 * 8,
                pad_token_id=1,
            ))

        with utils.Timer('copying and freezing weights'):
            prep_model(my_model, enru_model, layer_names_to_learn, do_learn_layer_norms=True)

        with utils.Timer('saving my_model'):
            my_model.save_pretrained(save_directory=constants.MODEL_DIR)
    else:
        with utils.Timer('loading my_model'):
            my_model = fsmt.MyFSMTForConditionalGeneration.from_pretrained(constants.MODEL_DIR, local_files_only=True)
            # freeze_weights(my_model, layers_to_skip=layer_names_to_learn, do_learn_layer_norms=True)

    data_collator = MySeq2SeqCollator(
        tokenizer=tok,
        model=my_model,
        padding='longest',
        label_pad_token_id=constants.INPUTS_PAD_ID,
    )

    from transformers import SchedulerType, TrainingArguments

    init_steps = 500

    trainer_args = Seq2SeqTrainingArguments(
        output_dir=constants.OUTPUT_DIR,  # output directory
        logging_dir=constants.LOGS_DIR,  # directory for storing logs
        # num_train_epochs=1,  # total # of training epochs
        max_steps=init_steps,
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=batch_size,  # batch size for evaluation
        warmup_steps=init_steps,    # number of warmup steps for learning rate scheduler
        learning_rate=1e-3,
        weight_decay=0.001,  # strength of weight decay
        predict_with_generate=True,
        sortish_sampler=True,
        do_eval=True,
        do_predict=True,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=100,
        dataloader_num_workers=4,
        report_to=['none'],
        lr_scheduler_type=SchedulerType.CONSTANT_WITH_WARMUP,
        logging_first_step=True,
        seed=seed,
    )

    # # round 1 -- train embeddings only
    # freeze_weights(my_model, layers_to_skip=layer_names_to_learn, do_learn_layer_norms=True)
    # trainer = Seq2SeqTrainer(
    #     model=my_model,
    #     args=trainer_args,
    #     train_dataset=ds_train,
    #     eval_dataset=ds_valid,
    #     tokenizer=tok,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    # )
    # train_out = trainer.train()
    # print(train_out)
    # unfreeze_weights(my_model)

    # round 2 -- train all weights
    trainer_args.max_steps = -1
    trainer_args.num_train_epochs = 1
    trainer_args.learning_rate = 1e-4
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
        # ignore_keys=[constants.DataNames.DECODER_INPUT_IDS, ]
    )
    print(predict_out)

    with tok.as_target_tokenizer():
        preds = tok.batch_decode(predict_out.predictions)

    print('Predictions:')
    for meta, pred_str, pred_tokens in zip(ds_valid.meta, preds, predict_out.predictions):
        print(meta['letters'], pred_str[:30], pred_tokens[:10])
