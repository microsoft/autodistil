# coding=utf-8
# 2020.08.28 - Changed regular fine-tuning to fine-tuning with adaptive width and depth
#              Huawei Technologies Co., Ltd <houlu3@huawei.com>
# Copyright (c) 2020, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright 2018 The Google AI Language Team Authors,  the HuggingFace Inc.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random, itertools
import math
import numpy as np
import torch
from collections import namedtuple
from tempfile import TemporaryDirectory
from pathlib import Path
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import  MSELoss

from transformers import (BertConfig, BertForSequenceClassification, BertForSequenceClassification_v1, BertForSequenceClassification_v1_AddHidLoss, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer)

from transformers import AdamW, WarmupLinearSchedule

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

import sys
import json

logger = logging.getLogger(__name__)
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return -torch.sum(targets_prob * student_likelihood, dim=-1).mean()


loss_mse = MSELoss()
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig)), ())
MODEL_CLASSES = {
    # 'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    # 'bert': (BertConfig, BertForSequenceClassification_v1, BertTokenizer),
    'bert': (BertConfig, BertForSequenceClassification_v1_AddHidLoss, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def batch_list_to_batch_tensors(features):
    batch = []

    for f in features:
        input_ids = f["input_ids"]
        num_tokens_a = f["num_tokens_a"][0]
        num_tokens_b = f["num_tokens_b"][0]
        masked_ids = f["masked_ids"]
        masked_pos = f["masked_pos"]
        masked_weights = f["masked_weights"]
        is_next = f["is_next"][0]
        task_idx = f["task_idx"][0]
        # num_tokens_a1 = f["num_tokens_a1"][0]

        num_tokens = num_tokens_a + num_tokens_b
        seq_len = len(input_ids)
        num_pads = seq_len - num_tokens

        # print("task:{}".format(task_idx))
        # print("qkv: {}".format(args.num_qkv))

        segment_ids = [0] * num_tokens_a + \
            [1] * num_tokens_b + [0] * num_pads
        input_mask = torch.tensor(
            [1] * num_tokens + [0] * num_pads, dtype=torch.long).unsqueeze(0).expand(seq_len, seq_len)

        mask_qkv = None
        batch.append((input_ids, segment_ids, input_mask, mask_qkv,
                      masked_ids, masked_pos, masked_weights, is_next, task_idx))

    batch_tensors = []
    for x in zip(*batch):
        if x[0] is None:
            batch_tensors.append(None)
        elif isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


def train(args, model, tokenizer, teacher_model=None, samples_per_epoch=None, num_data_epochs=None):
    """ Train the model """
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    # added from TinyBERT (DK)
    total_train_examples = 0
    for i in range(int(args.num_train_epochs)):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    num_train_optimization_steps = int(
        args.num_train_epochs_wholeset * total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    print("")
    print("torch.distributed.get_world_size(): ", torch.distributed.get_world_size())
    print("torch.cuda.device_count(): ", torch.cuda.device_count())
    print("")

    # train_sampler = RandomSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
    #                               batch_size=args.train_batch_size)
    # # train_dataloader_1 = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0, pin_memory=False, collate_fn=batch_list_to_batch_tensors)

    # t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    t_total = num_train_optimization_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    if args.model_type == 'roberta':
        args.warmup_steps = int(t_total*0.06)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)


    # if args.n_gpu > 1:
    #     print()
    #     print('DataParallel!')
    #     print()
    #     model = torch.nn.DataParallel(model)
    #     if teacher_model != None:
    #         teacher_model = torch.nn.DataParallel(teacher_model)

    # added from TinyBERT (DK)
    if args.local_rank != -1:
        print("DDP!!!")
        # try:
        #     from apex.parallel import DistributedDataParallel as DDP
        # except ImportError:
        #     raise ImportError(
        #         "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        if teacher_model != None:
            # teacher_model = DDP(teacher_model)
            teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        #model = DDP(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif n_gpu > 1:
        print("torch.nn.DataParallel!!!")
        model = torch.nn.DataParallel(model)
        if teacher_model != None:
            teacher_model = torch.nn.DataParallel(teacher_model)


    global_step = 0
    model.zero_grad()
    # train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)

    # loop over whole set (three epoch-data)
    # for epoch_wholeset in trange(int(args.num_train_epochs_wholeset), desc="Epoch_wholeset"):
    for epoch_wholeset in trange(int(args.num_train_epochs_wholeset), desc="Local_rank {}, Epoch_wholeset".format(args.local_rank)):
        # loop over each epoch-data
        # for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        for epoch in trange(int(args.num_train_epochs), desc="Local_rank {}, Epoch".format(args.local_rank)):
            epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                                num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory, output_cache_dir=args.output_cache_dir, args=args)
            if args.local_rank == -1:
                train_sampler = RandomSampler(epoch_dataset)
            else:
                train_sampler = DistributedSampler(epoch_dataset)
            train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

            tr_loss    = 0.0
            # hard_loss  = 0.0
            # logit_loss = 0.0
            # att_loss   = 0.0
            # rep_loss   = 0.0
            model.train()
            nb_tr_examples, nb_tr_steps = 0, 0
            # with tqdm(total=len(train_dataloader), desc="Epoch {}".format(epoch)) as pbar:
            with tqdm(total=len(train_dataloader), desc="Local_rank {}, Epoch {}".format(args.local_rank, epoch)) as pbar:
                # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                for step, batch in enumerate(tqdm(train_dataloader, desc="Local_rank {}, Iteration".format(args.local_rank), ascii=True)):
                    batch = tuple(t.to(args.device) for t in batch)

                    # # debug 
                    # if step == 10:
                    #     break

                    # current_best = 0
                    # output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')

                    # for _ in train_iterator:
                    #     epoch_iterator = tqdm(train_dataloader, desc="Iteration")
                    #     # epoch_iterator_1 = tqdm(train_dataloader_1, desc="Iteration")
                    #     for step, batch in enumerate(epoch_iterator):
                    #         print('step: ', step)
                    #         model.train()
                    #         batch = tuple(t.to(args.device) for t in batch)

                    inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3][:, 0],
                            'token_type_ids': batch[2] if args.model_type in ['bert'] else None}
                    inputs_stu = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3][:, 0],
                            'token_type_ids': batch[2] if args.model_type in ['bert'] else None, 'is_student': True}

                    print('')
                    print("Epoch_W {}, Epoch {}, Step {}, Local_rank {}, Labels {}".format(epoch_wholeset, epoch, step, args.local_rank, batch[0][0]))
                    print('')

                    # # prepare the hidden states and logits of the teacher model
                    # if args.training_phase == 'dynabertw' and teacher_model:
                    #     with torch.no_grad():
                    #         # teacher_logit: [32, 2]
                    #         # len(teacher_reps) = 13; teacher_reps[0]: [32, 128, 768]
                    #         # _, teacher_logit, teacher_reps, _, _ = teacher_model(**inputs)
                    #         _, teacher_logit, teacher_reps, teacher_atts, _ = teacher_model(**inputs)
                    
                    # prepare sub-nets
                    # subs_all = list(itertools.product(args.depth_mult_list, args.width_mult_list, args.hidden_mult_list))
                    subs_all = list(itertools.product(args.depth_mult_list, args.width_mult_list, args.hidden_mult_list, args.intermediate_mult_list))
                    # random.seed(int(global_step * torch.distributed.get_world_size())) ???
                    random.seed(int(global_step))                  
                    subs_sampled = [subs_all[-1], subs_all[0]] + random.sample(subs_all[1:-1], 2) # four subs: largest, smallest, two randomly sampled
                    # print('subs_sampled: ', subs_sampled)
                    # print('')
                    # print("Epoch_W {}, Epoch {}, Step {}, Local_rank {}, subs_sampled {}".format(epoch_wholeset, epoch, step, args.local_rank, subs_sampled))
                    # print('')

                    if args.training_phase == 'dynabert' and teacher_model:
                        hidden_max_all, logits_max_all = [], []
                        atts_max_all = []
                        # for width_mult in sorted(args.width_mult_list, reverse=True):
                        for hidden_mult in sorted(args.hidden_mult_list, reverse=True):
                            with torch.no_grad():
                                # teacher_logit: [32, 2]
                                # len(teacher_reps) = 13; teacher_reps[0]: [32, 128, 768]
                                # _, teacher_logit, teacher_reps, _, _ = teacher_model(**inputs)
                                _, teacher_logit, teacher_reps, teacher_atts, _ = teacher_model(**inputs)
                                hidden_max_all.append(teacher_reps)
                                logits_max_all.append(teacher_logit)
                                atts_max_all.append(teacher_atts)

                    # accumulate grads for all sampled sub-networks
                    for idx_sub in range(len(subs_sampled)):
                        # print('subs_sampled[idx_sub]: ', subs_sampled[idx_sub])
                        model.apply(lambda m: setattr(m, 'depth_mult', subs_sampled[idx_sub][0]))
                        
                        if args.training_phase == 'dynabert' or 'final_finetuning':
                            model = model.module if hasattr(model, 'module') else model
                            base_model = getattr(model, model.base_model_prefix, model)
                            n_layers = base_model.config.num_hidden_layers
                            depth = round(subs_sampled[idx_sub][0] * n_layers)
                            kept_layers_index = []
                            for i in range(depth):
                                kept_layers_index.append(math.floor(i / subs_sampled[idx_sub][0]))
                            kept_layers_index.append(n_layers)
                        
                        model.apply(lambda m: setattr(m, 'width_mult', subs_sampled[idx_sub][1]))
                        model.apply(lambda m: setattr(m, 'hidden_mult', subs_sampled[idx_sub][2]))
                        model.apply(lambda m: setattr(m, 'intermediate_mult', subs_sampled[idx_sub][3]))

                        # print('model: ', model)
                                
                        # stage 2: width- and depth- adaptive
                        if args.training_phase == 'dynabert':
                            if getattr(args, 'data_aug'):
                                loss, student_logit, student_reps, student_atts, _ = model(**inputs_stu)

                                # distillation loss of logits
                                if args.output_mode == "classification":
                                    # logit_loss = soft_cross_entropy(student_logit, logits_max_all[width_idx].detach())
                                    logit_loss = soft_cross_entropy(student_logit, teacher_logit.detach())
                                elif args.output_mode == "regression":
                                    logit_loss = 0

                                # # distillation loss of hidden states
                                # rep_loss = 0
                                # for student_rep, teacher_rep in zip(
                                #         student_reps, list(hidden_max_all[width_idx][i] for i in kept_layers_index)):
                                #     tmp_loss = loss_mse(student_rep, teacher_rep.detach())
                                #     rep_loss += tmp_loss
                                
                                # last layer
                                student_rep, teacher_rep = student_reps[-1], teacher_reps[-1]
                                rep_loss = loss_mse(student_rep, teacher_rep.detach())

                                # # distillation loss of attention
                                # att_loss = 0
                                # kept_layers_index
                                #   e.g., 
                                #       [0, 2, 4, 6, 8, 10, 12] for depth_mult = 0.5
                                #       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] for depth_mult = 1.0
                                
                                # last layer
                                # student_att, teacher_att = student_atts[-1], atts_max_all[width_idx][kept_layers_index[-2]] # '-2'??? check it again
                                student_att, teacher_att = student_atts[-1], teacher_atts[-1] 
                                student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(args.device),
                                                student_att)
                                teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(args.device),
                                                teacher_att)
                                att_loss = loss_mse(torch.sum(student_att, dim=1), torch.sum(teacher_att.detach(), dim=1))

                                # loss = args.depth_lambda1 * logit_loss + args.depth_lambda2 * rep_loss  # ground+truth and distillation
                                # logit_loss could represent hard_loss ???
                                # loss = args.depth_lambda1 * logit_loss + args.depth_lambda3 * att_loss
                                loss = args.depth_lambda1 * logit_loss + args.depth_lambda3 * att_loss + args.depth_lambda2 * rep_loss
                                # width_idx += 1  # move to the next width

                            else: 
                                hard_loss, student_logit, student_reps, student_atts, _ = model(**inputs_stu)

                                # print('')
                                # print("Epoch_W {}, Epoch {}, Step {}, Local_rank {}, Sub {}, Student_reps {}".format(epoch_wholeset, epoch, step, args.local_rank, idx_sub, student_reps[2][0]))
                                # print('')

                                # distillation loss of logits
                                if args.output_mode == "classification":
                                    # logit_loss = soft_cross_entropy(student_logit, logits_max_all[width_idx].detach())
                                    logit_loss = soft_cross_entropy(student_logit, teacher_logit.detach())
                                elif args.output_mode == "regression":
                                    logit_loss = 0

                                # # distillation loss of hidden states
                                # rep_loss = 0
                                # for student_rep, teacher_rep in zip(
                                #         student_reps, list(hidden_max_all[width_idx][i] for i in kept_layers_index)):
                                #     tmp_loss = loss_mse(student_rep, teacher_rep.detach())
                                #     rep_loss += tmp_loss
                                
                                # last layer
                                student_rep, teacher_rep = student_reps[-1], teacher_reps[-1]
                                rep_loss = loss_mse(student_rep, teacher_rep.detach())

                                ##  distillation loss of attention
                                # att_loss = 0
                                
                                # last layer
                                # student_att: [32, width_mult*12, 128, 128]
                                # teacher_att: [32, 12, 128, 128]
                                # student_att, teacher_att = student_atts[-1], atts_max_all[width_idx][kept_layers_index[-2]] # '-2'??? check it again
                                student_att, teacher_att = student_atts[-1], teacher_atts[-1]
                                student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(args.device),
                                                student_att)
                                teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(args.device),
                                                teacher_att)
                                att_loss = loss_mse(torch.sum(student_att, dim=1), torch.sum(teacher_att.detach(), dim=1))

                                # loss = args.depth_lambda1 * logit_loss + args.depth_lambda2 * rep_loss  # ground+truth and distillation
                                # logit_loss could represent hard_loss ???
                                # loss = args.depth_lambda4 * hard_loss + args.depth_lambda1 * logit_loss + args.depth_lambda3 * att_loss
                                loss = args.depth_lambda4 * hard_loss + args.depth_lambda1 * logit_loss + args.depth_lambda3 * att_loss + args.depth_lambda2 * rep_loss
                                # width_idx += 1  # move to the next width

                                # if (global_step + 1) % args.printloss_step == 0:
                                #     print("------------Global_step {}----------".format(global_step))
                                #     print("total loss", loss.item())
                                #     print("hard_loss", hard_loss.item())
                                #     print("logit_loss", logit_loss.item())
                                #     print("att_loss", att_loss.item())
                                #     print("rep_loss", rep_loss.item())

                        # stage 3: final finetuning
                        else:
                            loss = model(**inputs_stu)[0]

                        if global_step % 100 == 0:
                            # print('')
                            # # print('local_rank, loss: ', args.local_rank, loss)
                            # # print("Local_rank {}, Loss {}".format(args.local_rank, loss))
                            print("Epoch_W {}, Epoch {}, Step {}, Local_rank {}, Sub {}, Loss {}".format(epoch_wholeset, epoch, step, args.local_rank, idx_sub, loss))
                            print('')

                        if args.n_gpu > 1:
                            loss = loss.mean()
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps

                        # loss.backward()
                        
                        # added from TinyBERT (DK)
                        if args.fp16:
                            optimizer.backward(loss)
                        else:
                            loss.backward()
                    
                    # clip the accumulated grad from all widths
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    tr_loss += loss.item()
                    nb_tr_steps += 1
                    pbar.update(1)

                    mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        model.zero_grad()
                        global_step += 1

                        # # save
                        # if global_step > 0 and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        #     logger.info("Saving model checkpoint to %s", args.output_dir)
                        #     model_to_save = model.module if hasattr(model, 'module') else model
                        #     model_to_save.save_pretrained(args.output_dir)
                        #     torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
                        #     model_to_save.config.to_json_file(os.path.join(args.output_dir, CONFIG_NAME))
                        #     tokenizer.save_vocabulary(args.output_dir)


                        # # evaluate
                        # if global_step > 0 and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        #     print('Evaluate: global_step is ', global_step)
                        #     if args.evaluate_during_training:
                        #         acc = []
                        #         if args.task_name == "mnli":   # for both MNLI-m and MNLI-mm
                        #             acc_both = []

                        #         # collect performance of all sub-networks
                        #         for depth_mult in sorted(args.depth_mult_list, reverse=True):
                        #             model.apply(lambda m: setattr(m, 'depth_mult', depth_mult))
                        #             for width_mult in sorted(args.width_mult_list, reverse=True):
                        #                 model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                        #                 for hidden_mult in sorted(args.hidden_mult_list, reverse=True):
                        #                     model.apply(lambda m: setattr(m, 'hidden_mult', hidden_mult))

                        #                     results = evaluate(args, model, tokenizer)
                        #                     # print("results: ", results)

                        #                     logger.info("********** start evaluate results *********")
                        #                     logger.info("depth_mult: %s ", depth_mult)
                        #                     logger.info("width_mult: %s ", width_mult)
                        #                     logger.info("hidden_mult: %s ", hidden_mult)
                        #                     logger.info("results: %s ", results)
                        #                     logger.info("********** end evaluate results *********")

                        #                     acc.append(list(results.values())[0])
                        #                     if args.task_name == "mnli":
                        #                         acc_both.append(list(results.values())[0:2])

                                # # save model
                                # if sum(acc) > current_best:
                                #     current_best = sum(acc)
                                #     if args.task_name == "mnli":
                                #         print("***best***{}\n".format(acc_both))
                                #         with open(output_eval_file, "a") as writer:
                                #             writer.write("{}\n".format(acc_both))
                                #     else:
                                #         print("***best***{}\n".format(acc))
                                #         with open(output_eval_file, "a") as writer:
                                #             writer.write("{}\n" .format(acc))

                                #     logger.info("Saving model checkpoint to %s", args.output_dir)
                                #     model_to_save = model.module if hasattr(model, 'module') else model
                                #     model_to_save.save_pretrained(args.output_dir)
                                #     torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
                                #     model_to_save.config.to_json_file(os.path.join(args.output_dir, CONFIG_NAME))
                                #     tokenizer.save_vocabulary(args.output_dir)
            
        # save after each epoch of whole data
        
        # logger.info("Saving model checkpoint to %s", args.output_dir)
        # model_to_save = model.module if hasattr(model, 'module') else model
        # model_to_save.save_pretrained(args.output_dir)
        # torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        # model_to_save.config.to_json_file(os.path.join(args.output_dir, CONFIG_NAME))
        # tokenizer.save_vocabulary(args.output_dir)

        # model_name = "epoch_{}_{}".format(epoch_wholeset, WEIGHTS_NAME)
        # logging.info("** ** * Saving fine-tuned model ** ** * ")
        # model_to_save = model.module if hasattr(model, 'module') else model
        # output_model_file = os.path.join(args.output_dir, model_name)
        # output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        # torch.save(model_to_save.state_dict(), output_model_file)
        # model_to_save.config.to_json_file(output_config_file)
        # tokenizer.save_vocabulary(args.output_dir)
        # torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # if args.local_rank % torch.cuda.device_count() == 0:
        if args.local_rank in [-1, 0]:
            print('')
            print("Saving model checkpoint via GPU (local_rank) = {}".format(args.local_rank))
            print('')

            if not os.path.exists(os.path.join(args.output_dir, "epoch_{}/".format(epoch_wholeset))):
                os.makedirs(os.path.join(args.output_dir, "epoch_{}/".format(epoch_wholeset)))

            model_name = "epoch_{}/{}".format(epoch_wholeset, WEIGHTS_NAME)
            config_name = "epoch_{}/{}".format(epoch_wholeset, CONFIG_NAME)

            logging.info("** ** * Saving fine-tuned model ** ** * ")
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(args.output_dir, model_name)
            output_config_file = os.path.join(args.output_dir, config_name)
            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)

            tokenizer.save_vocabulary(os.path.join(args.output_dir, "epoch_{}/".format(epoch_wholeset)))
            torch.save(args, os.path.join(args.output_dir, "epoch_{}/".format(epoch_wholeset), 'training_args.bin'))


def evaluate(args, model, tokenizer, prefix=""):
    """ Evaluate the model """
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + 'MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert'] else None
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        if eval_task == 'mnli-mm':
            results.update({'acc_mm':result['acc']})
        else:
            results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt") # wirte all the results to the same file
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            writer.write("\n")
    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):

    processor = processors[task]()
    output_mode = output_modes[task]
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()
    if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
        label_list[1], label_list[2] = label_list[2], label_list[1]
    examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
    if not evaluate and args.data_aug:
        examples_aug = processor.get_train_examples_aug(args.data_dir)
        examples = examples + examples_aug
    features = convert_examples_to_features(examples,
                                            tokenizer,
                                            label_list=label_list,
                                            max_length=args.max_seq_length,
                                            output_mode=output_mode,
                                            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def compute_neuron_head_importance(args, model, tokenizer):
    """ This method shows how to compute:
        - neuron importance scores based on loss according to http://arxiv.org/abs/1905.10650
    """
    # prepare things for heads
    model = model.module if hasattr(model, 'module') else model
    base_model = getattr(model, model.base_model_prefix, model)
    n_layers, n_heads = base_model.config.num_hidden_layers, base_model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)
    head_mask = torch.ones(n_layers, n_heads).to(args.device)
    head_mask.requires_grad_(requires_grad=True)

    # collect weights
    intermediate_weight = []
    intermediate_bias = []
    output_weight = []
    for name, w in model.named_parameters():
        if 'intermediate' in name:
            if w.dim() > 1:
                intermediate_weight.append(w)
            else:
                intermediate_bias.append(w)

        if 'output' in name and 'attention' not in name:
            if w.dim() > 1:
                output_weight.append(w)

    neuron_importance = []
    for w in intermediate_weight:
        neuron_importance.append(torch.zeros(w.shape[0]).to(args.device))

    model.to(args.device)

    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + 'MM') if args.task_name == "mnli" else (args.output_dir,)

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, _, label_ids = batch
            segment_ids = batch[2] if args.model_type=='bert' else None  # RoBERTa does't use segment_ids

            # calculate head importance
            outputs = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids,
                            head_mask=head_mask)
            loss = outputs[0]
            loss.backward()
            head_importance += head_mask.grad.abs().detach()

            # calculate  neuron importance
            for w1, b1, w2, current_importance in zip(intermediate_weight, intermediate_bias, output_weight, neuron_importance):
                current_importance += ((w1 * w1.grad).sum(dim=1) + b1 * b1.grad).abs().detach()
                current_importance += ((w2 * w2.grad).sum(dim=0)).abs().detach()

    return head_importance, neuron_importance


def reorder_neuron_head(model, head_importance, neuron_importance):
    """ reorder neurons based on their importance.

        Arguments:
            model: bert model
            head_importance: 12*12 matrix for head importance in 12 layers
            neuron_importance: list for neuron importance in 12 layers.
    """
    model = model.module if hasattr(model, 'module') else model
    base_model = getattr(model, model.base_model_prefix, model)

    # reorder heads and ffn neurons
    for layer, current_importance in enumerate(neuron_importance):
        # reorder heads
        idx = torch.sort(head_importance[layer], descending=True)[-1]
        base_model.encoder.layer[layer].attention.reorder_heads(idx)
        # reorder neurons
        idx = torch.sort(current_importance, descending=True)[-1]
        base_model.encoder.layer[layer].intermediate.reorder_neurons(idx)
        base_model.encoder.layer[layer].output.reorder_neurons(idx)

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")

def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    if len(tokens) > max_seq_length:
        logger.info('len(tokens): {}'.format(len(tokens)))
        logger.info('tokens: {}'.format(tokens))
        tokens = tokens[:max_seq_length]

    if len(tokens) != len(segment_ids):
        logger.info('tokens: {}\nsegment_ids: {}'.format(tokens, segment_ids))
        segment_ids = [0] * len(tokens)

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    # lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=0)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next)
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False, output_cache_dir=None, args=None):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = int(epoch % num_data_epochs)
        logger.info('training_path: {}'.format(training_path))
        data_file = training_path / "epoch_{}.json".format(self.data_epoch)
        metrics_file = training_path / "epoch_{}_metrics.json".format(self.data_epoch)

        logger.info('data_file: {}'.format(data_file))
        logger.info('metrics_file: {}'.format(metrics_file))

        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            # self.working_dir = Path('/cache')
            # self.working_dir = Path('/home/jiachao/data/DongkuanXu/TinyBERT/General_KD/cache')
            self.working_dir = Path(output_cache_dir)
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir/'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
            is_nexts = np.memmap(filename=self.working_dir/'is_nexts.memmap',
                                 shape=(num_samples,), mode='w+', dtype=np.bool)
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)

        logging.info("Loading training examples for epoch {}".format(epoch))

        with data_file.open() as f:
            # for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Local_rank {}, Training examples".format(args.local_rank))):
                
                if i == 1000:
                    break
                if i % 100000 == 0:
                    print("")
                    # print("local_rank, line i: ", args.local_rank, i)
                    print("Local_rank {}, line {}".format(args.local_rank, i))
                    print("")

                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next

        # assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        # self.lm_label_ids = lm_label_ids
        self.lm_label_ids = lm_label_ids * 0
        self.is_nexts = is_nexts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(int(self.is_nexts[item])))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="The student (and teacher) model dir.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where trained model is saved.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--evaluate_during_training", default=True,
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=1000,
                        help="Log every X updates steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
                        help="dropout rate on hidden states.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float,
                        help="dropout rate on attention probs.")

    parser.add_argument('--data_aug', action='store_true', help="whether using data augmentation")
    # for depth direction
    parser.add_argument('--depth_mult_list', type=str, default='1.',
                        help="the possible depths used for training, e.g., '1.' is for default")
    parser.add_argument("--depth_lambda1", default=1.0, type=float,
                        help="logit matching coef.")
    parser.add_argument("--depth_lambda2", default=1.0, type=float,
                        help="hidden states matching coef.")
    parser.add_argument("--depth_lambda3", default=0.001, type=float,
                        help="attention matching coef.")
    parser.add_argument("--depth_lambda4", default=1.0, type=float,
                        help="hard label matching coef.")
    # for width direction (changing hidden_dim)
    parser.add_argument('--width_mult_list', type=str, default='1.',
                        help="the possible widths used for training, e.g., '1.' is for separate training "
                             "while '0.25,0.5,0.75,1.0' is for vanilla slimmable training")
    parser.add_argument("--width_lambda1", default=1.0, type=float,
                        help="logit matching coef.")
    parser.add_argument("--width_lambda2", default=1.0, type=float,
                        help="hidden states matching coef.")
    parser.add_argument("--width_lambda3", default=0.001, type=float,
                        help="attention matching coef.")
    parser.add_argument("--width_lambda4", default=1.0, type=float,
                        help="hard label matching coef.")

    parser.add_argument("--training_phase", default="dynabertw", type=str,
                        help="can be finetuning, dynabertw, dynabert, final_finetuning")

    parser.add_argument('--printloss_step', type=int, default=100)

    # for hidden_dim direction
    parser.add_argument('--hidden_mult_list', type=str, default='1.',
                        help="the possible depths used for training, e.g., '1.' is for default")

    parser.add_argument("--pregenerated_data",
                        type=Path,
                        required=True)
    parser.add_argument("--reduce_memory",
                        action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--output_cache_dir", default=None, type=str, required=True,
                        help="The output directory")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--num_train_epochs_wholeset", default=3.0, type=float,
                        help="Total number of training epochs to perform.")

    # for intermediate ratio direction
    parser.add_argument('--intermediate_mult_list', type=str, default='1.',
                        help="the possible intermediate size used for training, e.g., '1.' is for default")

    args = parser.parse_args()

    args.width_mult_list = [float(width) for width in args.width_mult_list.split(',')]
    args.depth_mult_list = [float(depth) for depth in args.depth_mult_list.split(',')]
    args.hidden_mult_list = [float(hidden) for hidden in args.hidden_mult_list.split(',')]
    args.intermediate_mult_list = [float(intermediate) for intermediate in args.intermediate_mult_list.split(',')]

    # added from TinyBERT (DK)
    samples_per_epoch = []
    for i in range(int(args.num_train_epochs)):
        epoch_file = args.pregenerated_data / "epoch_{}.json".format(i)
        metrics_file = args.pregenerated_data / "epoch_{}_metrics.json".format(i)
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                exit("No training data was found!")
            print("Warning! There are fewer epochs of pregenerated data ({}) than training epochs ({}).".format(i, args.num_train_epochs))
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break
    # ??? (DK)
    else:
        num_data_epochs = args.num_train_epochs # num_data_epochs should be 3 and it is used for processing each subset of wiki

    print('args.local_rank: ', args.local_rank)

    # Setup CUDA, GPU & distributed training
    # device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    # args.n_gpu = torch.cuda.device_count()
    # print('n_gpu: ', args.n_gpu)
    # args.device = device

    # added from TinyBERT (DK)
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    #                     datefmt='%m/%d/%Y %H:%M:%S',
    #                     level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    args.n_gpu = n_gpu
    args.device = device


    # Set seed
    set_seed(args)

    # Prepare GLUE task: provide num_labels here
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    # num_labels = len(label_list)
    num_labels = len(label_list)

    # prepare model, tokernizer and config
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_dir, num_labels=num_labels, finetuning_task=args.task_name)
    # config = config_class.from_pretrained(args.model_dir, finetuning_task=args.task_name)
    # print('config: ', config)
    # print('')
    # config.num_labels = num_labels
    # print('config.hidden_size: ', config.hidden_size)
    config.output_attentions, config.output_hidden_states, config.output_intermediate = True,True,True
    tokenizer = tokenizer_class.from_pretrained(args.model_dir, do_lower_case=args.do_lower_case)

    # load teacher model if necessary
    if args.training_phase == 'dynabertw' or args.training_phase == 'dynabert':
        teacher_model = model_class.from_pretrained(args.model_dir, config=config)
        teacher_model.to(args.device)
    else:
        teacher_model = None

    # load student model if necessary
    model = model_class.from_pretrained(args.model_dir, config=config)

    # # print('model: ', model)
    # s = model.state_dict()
    # s = {k:v for k,v in s.items() if not k.startswith('classifier')}
    # config.num_labels = len(label_list)
    # model = model_class.from_pretrained(None, config=config, state_dict=s)
    # # print('model: ', model)

    # # if args.training_phase == 'dynabertw':
    # if args.training_phase == 'dynabertw' or args.training_phase == 'dynabert':
    #     print('Network rewiring starts')
    #     # rewire the network according to the importance of attention heads and neurons
    #     head_importance, neuron_importance = compute_neuron_head_importance(args, model, tokenizer)
    #     reorder_neuron_head(model, head_importance, neuron_importance)
    #     print('Network rewiring done!')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # print('')
        # print('Constructing train_dataset starts')
        # train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        # print('Constructing train_dataset done!')
        # print('')
        
        if teacher_model:
            # global_step, tr_loss = train(args, train_dataset, model, tokenizer, teacher_model, samples_per_epoch, num_data_epochs)
            train(args, model, tokenizer, teacher_model, samples_per_epoch, num_data_epochs)
        else:
            train(args, model, tokenizer, samples_per_epoch, num_data_epochs)

        # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    
    # sys.argv.extend([
    #     '--model_type', 'bert',
    #     '--task_name', 'MRPC',
    #     '--do_train',
    #     '--data_dir', '/home/t-dongkuanxu/Pretrained-Language-Model/Data_and_Models/Data_GLUE/glue_data/MRPC/',
    #     '--model_dir', '/home/t-dongkuanxu/Pretrained-Language-Model/Data_and_Models/Local_models/finetuned_BERTs/bert_base_uncased_MRPC/',
    #     '--output_dir', '/home/t-dongkuanxu/Pretrained-Language-Model/DynaBERT/output_debug/',
    #     '--max_seq_length', '128',
    #     '--learning_rate', '2e-5',
    #     '--per_gpu_train_batch_size', '32',
    #     '--num_train_epochs', '1',
    #     '--width_mult_list', '1.0',
    #     '--depth_mult_list', '0.25,0.5,1.0',
    #     '--hidden_mult_list', '0.25,0.5,1.0',
    #     '--width_lambda1', '1.0',
    #     '--width_lambda2', '1.0',
    #     '--width_lambda3', '0.001',
    #     '--width_lambda4', '1.0',
    #     '--depth_lambda1', '1.0',
    #     '--depth_lambda2', '1.0',
    #     '--depth_lambda3', '0.001',
    #     '--depth_lambda4', '1.0',
    #     '--training_phase', 'dynabert',
    #     '--logging_steps', '20',
    #     '--printloss_step', '10',
    #     ])

    # sys.argv.extend([
    #     '--model_type', 'bert',
    #     '--task_name', 'MRPC',
    #     '--do_train',
    #     '--data_dir', '/home/t-dongkuanxu/Pretrained-Language-Model/Data_and_Models/Data_GLUE/glue_data/MRPC/',
    #     '--model_dir', '/home/t-dongkuanxu/Pretrained-Language-Model/Data_and_Models/Local_models/pretrained_BERTs/BERT_base_uncased/',
    #     '--output_dir', '/home/t-dongkuanxu/Pretrained-Language-Model/DynaBERT/output_debug/',
    #     '--max_seq_length', '128',
    #     '--learning_rate', '2e-5',
    #     '--per_gpu_train_batch_size', '32',
    #     '--num_train_epochs', '3',
    #     '--depth_mult_list', '0.25,0.33333,0.5',
    #     '--width_mult_list', '0.5,0.66667,1.0',
    #     '--hidden_mult_list', '0.25,0.5,0.75,1.0',
    #     '--depth_lambda1', '1.0',
    #     '--depth_lambda2', '1.0',
    #     '--depth_lambda3', '0.001',
    #     '--depth_lambda4', '0.0',
    #     '--training_phase', 'dynabert',
    #     '--logging_steps', '20',
    #     '--printloss_step', '10',
    #     '--weight_decay', '0.01',
    #     '--output_cache_dir', '/home/t-dongkuanxu/Pretrained-Language-Model/Data_and_Models/Outputs/General_KD/cache/',
    #     '--pregenerated_data', '/home/t-dongkuanxu/Pretrained-Language-Model/Data_and_Models/English_Wiki/corpus_jsonfile_for_general_KD/',
    #     ])

    main()
