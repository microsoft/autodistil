# coding=utf-8
# 2020.08.28 - (1) Changed fixed sized Transformer layer to support adptive width; and
#              (2) Added modules to rewire the BERT model according to the importance of attention
#                  heads and neurons in the intermediate layer of Feed-forward Network.
#              Huawei Technologies Co., Ltd <houlu3@huawei.com>
# Copyright (c) 2020, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and the HuggingFace Inc. team.
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
"""PyTorch BERT model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os
import sys
sys.path.append("..")

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from .modeling_utils import PreTrainedModel
from .configuration_bert import BertConfig

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    'bert-base-german-dbmdz-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin",
    'bert-base-german-dbmdz-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin",
}


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}
BertLayerNorm = torch.nn.LayerNorm


def round_to_nearest(input_size, width_mult, num_heads, min_value=1):
    new_width_mult = round(num_heads * width_mult)*1.0/num_heads
    input_size = int(new_width_mult * input_size)
    new_input_size = max(min_value, input_size)
    return new_input_size


class DynaLinear(nn.Linear):
    def __init__(self, in_features, out_features, num_heads, bias=True, dyna_dim=[True, True]):
        super(DynaLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features
        self.num_heads = num_heads
        self.width_mult = 1.
        self.dyna_dim = dyna_dim

    def forward(self, input):
        if self.dyna_dim[0]:
            self.in_features = round_to_nearest(self.in_features_max, self.width_mult, self.num_heads)
        if self.dyna_dim[1]:
            self.out_features = round_to_nearest(self.out_features_max, self.width_mult, self.num_heads)
        
        weight = self.weight[:self.out_features, :self.in_features]

        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class DynaLinear_v1(nn.Linear):
    def __init__(self, in_features, out_features, num_heads, bias=True, dyna_dim=[True, True], dyna_cate=['hid', 'head']):
        super(DynaLinear_v1, self).__init__(
            in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features
        self.num_heads = num_heads
        self.width_mult = 1.
        self.dyna_dim = dyna_dim

        self.dyna_cate = dyna_cate
        self.hidden_mult = 1.

        self.intermediate_mult = 4.

    def forward(self, input):
        if self.dyna_dim[0]:
            # self.in_features = round_to_nearest(self.in_features_max, self.width_mult, self.num_heads)
            if self.dyna_cate[0] == 'head':
                self.in_features = round_to_nearest(self.in_features_max, self.width_mult, self.num_heads)
            if self.dyna_cate[0] == 'hid':
                self.in_features = int(self.in_features_max * self.hidden_mult)
            if self.dyna_cate[0] == 'inter':
                self.in_features = int(self.in_features_max * self.hidden_mult * self.intermediate_mult * 0.25) # because originally intermediate = 4 * hidden 

        if self.dyna_dim[1]:
            # self.out_features = round_to_nearest(self.out_features_max, self.width_mult, self.num_heads)
            if self.dyna_cate[1] == 'head':
                self.out_features = round_to_nearest(self.out_features_max, self.width_mult, self.num_heads)
            if self.dyna_cate[1] == 'hid':
                self.out_features = int(self.out_features_max * self.hidden_mult)
            if self.dyna_cate[1] == 'inter':
                self.out_features = int(self.out_features_max * self.hidden_mult * self.intermediate_mult * 0.25)

        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class DynaEmbedding_v1(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False, _weight=None, dyna_dim=[True, True], dyna_cate=['embd', 'hid']):
        super(DynaEmbedding_v1, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=_weight)
        # self.num_embeddings = num_embeddings
        # self.embedding_dim = embedding_dim
        self.num_embeddings_max = num_embeddings
        self.embedding_dim_max = embedding_dim

        self.width_mult = 1.
        self.hidden_mult = 1.
        self.dyna_dim = dyna_dim
        self.dyna_cate = dyna_cate

    def forward(self, input):
        # for v1, the num_embeddings is fixed
        if self.dyna_dim[1]:
            if self.dyna_cate[1] == 'hid':
                self.embedding_dim = int(self.embedding_dim_max * self.hidden_mult)

        weight = self.weight[:self.num_embeddings, :self.embedding_dim]

        return nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


class DynaBertLayerNorm_v1(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, dyna_dim=[True], dyna_cate=['hid']):
        super(DynaBertLayerNorm_v1, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.normalized_shape_max = normalized_shape
        self.hidden_mult = 1.
        self.dyna_dim = dyna_dim
        self.dyna_cate = dyna_cate

    def forward(self, input):
        # only one dim for layernorm
        # print('self.normalized_shape_max: ', self.normalized_shape_max)
        # print('self.hidden_mult: ', self.hidden_mult)
        if self.dyna_dim[0]:
            if self.dyna_cate[0] == 'hid':
                self.normalized_shape = int(self.normalized_shape_max * self.hidden_mult)

        if self.weight is not None:
            weight = self.weight[:self.normalized_shape]
        else:
            weight = self.weight
        
        if self.bias is not None:
            bias = self.bias[:self.normalized_shape]
        else:
            bias = self.bias

        return nn.functional.layer_norm(
            input, [self.normalized_shape], weight, bias, self.eps)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertEmbeddings_v1(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings_v1, self).__init__()

        # self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.word_embeddings = DynaEmbedding_v1(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = DynaEmbedding_v1(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = DynaEmbedding_v1(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        # self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm = DynaBertLayerNorm_v1(config.hidden_size, eps=config.layer_norm_eps, dyna_dim=[True], dyna_cate=['hid'])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.orig_num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # dense layer for adaptive width
        self.query = DynaLinear(config.hidden_size, self.all_head_size, config.num_attention_heads, dyna_dim=[False, True])
        self.key = DynaLinear(config.hidden_size, self.all_head_size, config.num_attention_heads, dyna_dim=[False, True])
        self.value = DynaLinear(config.hidden_size, self.all_head_size, config.num_attention_heads, dyna_dim=[False, True])

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        self.num_attention_heads = round(self.orig_num_attention_heads * self.query.width_mult)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # output attention scores when needed
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs


# class BertSelfAttention_v1(nn.Module):
#     def __init__(self, config):
#         super(BertSelfAttention_v1, self).__init__()
#         if config.hidden_size % config.num_attention_heads != 0:
#             raise ValueError(
#                 "The hidden size (%d) is not a multiple of the number of attention "
#                 "heads (%d)" % (config.hidden_size, config.num_attention_heads))
#         self.output_attentions = config.output_attentions

#         self.num_attention_heads = config.num_attention_heads
#         self.orig_num_attention_heads = config.num_attention_heads
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size
        
#         # dense layer for adaptive width
#         # self.query = DynaLinear(config.hidden_size, self.all_head_size, config.num_attention_heads, dyna_dim=[False, True])
#         self.query = DynaLinear_v1(config.hidden_size, self.all_head_size, config.num_attention_heads, dyna_dim=[True, True], dyna_cate=['hid', 'head'])
#         self.key = DynaLinear_v1(config.hidden_size, self.all_head_size, config.num_attention_heads, dyna_dim=[True, True], dyna_cate=['hid', 'head'])
#         self.value = DynaLinear_v1(config.hidden_size, self.all_head_size, config.num_attention_heads, dyna_dim=[True, True], dyna_cate=['hid', 'head'])

#         self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(self, hidden_states, attention_mask=None, head_mask=None):
#         mixed_query_layer = self.query(hidden_states) # [batch, length, hid]
#         mixed_key_layer = self.key(hidden_states)
#         mixed_value_layer = self.value(hidden_states)

#         self.num_attention_heads = round(self.orig_num_attention_heads * self.query.width_mult)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         query_layer = self.transpose_for_scores(mixed_query_layer) # [batch, head, length, head_size]
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [batch, head, length, length]
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         if attention_mask is not None:
#             # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
#             attention_scores = attention_scores + attention_mask

#         # Normalize the attention scores to probabilities.
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)

#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         attention_probs = self.dropout(attention_probs)

#         # Mask heads if we want to
#         if head_mask is not None:
#             attention_probs = attention_probs * head_mask

#         context_layer = torch.matmul(attention_probs, value_layer)

#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)

#         # output attention scores when needed
#         outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
#         return outputs


class BertSelfAttention_v1(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention_v1, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.orig_num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # dense layer for adaptive width
        # self.query = DynaLinear(config.hidden_size, self.all_head_size, config.num_attention_heads, dyna_dim=[False, True])
        self.query = DynaLinear_v1(config.hidden_size, self.all_head_size, config.num_attention_heads, dyna_dim=[True, True], dyna_cate=['hid', 'head'])
        self.key = DynaLinear_v1(config.hidden_size, self.all_head_size, config.num_attention_heads, dyna_dim=[True, True], dyna_cate=['hid', 'head'])
        self.value = DynaLinear_v1(config.hidden_size, self.all_head_size, config.num_attention_heads, dyna_dim=[True, True], dyna_cate=['hid', 'head'])

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.num_attention_heads_teacher = 12
        self.attention_head_size_adjusted = self.attention_head_size

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_for_KD(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads_teacher, self.attention_head_size_adjusted)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states) # [batch, length, 'hid'], 'hid' = width_mult * hid
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        self.num_attention_heads = round(self.orig_num_attention_heads * self.query.width_mult)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        query_layer = self.transpose_for_scores(mixed_query_layer) # [batch, 'head', length, head_size], 'head' = width_mult * head
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [batch, 'head', length, length]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # added by DK
        self.attention_head_size_adjusted = self.all_head_size // self.num_attention_heads_teacher
        all_head_size_round = int(self.attention_head_size_adjusted * self.num_attention_heads_teacher)
        query_layer_KD = self.transpose_for_scores_for_KD(mixed_query_layer[:,:,:all_head_size_round]) # [batch, 12, length, 'head_size']
        key_layer_KD = self.transpose_for_scores_for_KD(mixed_key_layer[:,:,:all_head_size_round])
        value_layer_KD = self.transpose_for_scores_for_KD(mixed_value_layer[:,:,:all_head_size_round])

        attention_scores_QQ = torch.matmul(query_layer_KD, query_layer_KD.transpose(-1, -2)) # [batch, 12, length, length]
        attention_scores_KK = torch.matmul(key_layer_KD, key_layer_KD.transpose(-1, -2))
        attention_scores_VV = torch.matmul(value_layer_KD, value_layer_KD.transpose(-1, -2))
        attention_scores_QQ = attention_scores_QQ / math.sqrt(self.attention_head_size_adjusted)
        attention_scores_KK = attention_scores_KK / math.sqrt(self.attention_head_size_adjusted)
        attention_scores_VV = attention_scores_VV / math.sqrt(self.attention_head_size_adjusted)        

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores_QQ = attention_scores_QQ + attention_mask
            attention_scores_KK = attention_scores_KK + attention_mask
            attention_scores_VV = attention_scores_VV + attention_mask


        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # output attention scores when needed
        # outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        outputs = (context_layer, attention_scores, attention_scores_QQ, attention_scores_KK, attention_scores_VV) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        # dense layer for adaptive width
        self.dense = DynaLinear(config.hidden_size, config.hidden_size, config.num_attention_heads, dyna_dim=[True, False])
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertSelfOutput_v1(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput_v1, self).__init__()
        # dense layer for adaptive width
        # self.dense = DynaLinear(config.hidden_size, config.hidden_size, config.num_attention_heads, dyna_dim=[True, False])
        self.dense = DynaLinear_v1(config.hidden_size, config.hidden_size, config.num_attention_heads, dyna_dim=[True, True], dyna_cate=['head', 'hid'])
        
        # self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm = DynaBertLayerNorm_v1(config.hidden_size, eps=config.layer_norm_eps, dyna_dim=[True], dyna_cate=['hid'])
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def reorder_heads(self, idx):
        n, a = self.self.num_attention_heads, self.self.attention_head_size
        index = torch.arange(n*a).reshape(n, a)[idx].view(-1).contiguous().long()

        def reorder_head_matrix(linearLayer, index, dim=0):
            index = index.to(linearLayer.weight.device)
            W = linearLayer.weight.index_select(dim, index).clone().detach()
            if linearLayer.bias is not None:
                if dim == 1:
                    b = linearLayer.bias.clone().detach()
                else:
                    b = linearLayer.bias[index].clone().detach()

            linearLayer.weight.requires_grad = False
            linearLayer.weight.copy_(W.contiguous())
            linearLayer.weight.requires_grad = True
            if linearLayer.bias is not None:
                linearLayer.bias.requires_grad = False
                linearLayer.bias.copy_(b.contiguous())
                linearLayer.bias.requires_grad = True

        reorder_head_matrix(self.self.query, index)
        reorder_head_matrix(self.self.key, index)
        reorder_head_matrix(self.self.value, index)
        reorder_head_matrix(self.output.dense, index, dim=1)

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertAttention_v1(nn.Module):
    def __init__(self, config):
        super(BertAttention_v1, self).__init__()
        self.self = BertSelfAttention_v1(config)
        self.output = BertSelfOutput_v1(config)

    def reorder_heads(self, idx):
        n, a = self.self.num_attention_heads, self.self.attention_head_size
        index = torch.arange(n*a).reshape(n, a)[idx].view(-1).contiguous().long()

        def reorder_head_matrix(linearLayer, index, dim=0):
            index = index.to(linearLayer.weight.device)
            W = linearLayer.weight.index_select(dim, index).clone().detach()
            if linearLayer.bias is not None:
                if dim == 1:
                    b = linearLayer.bias.clone().detach()
                else:
                    b = linearLayer.bias[index].clone().detach()

            linearLayer.weight.requires_grad = False
            linearLayer.weight.copy_(W.contiguous())
            linearLayer.weight.requires_grad = True
            if linearLayer.bias is not None:
                linearLayer.bias.requires_grad = False
                linearLayer.bias.copy_(b.contiguous())
                linearLayer.bias.requires_grad = True

        reorder_head_matrix(self.self.query, index)
        reorder_head_matrix(self.self.key, index)
        reorder_head_matrix(self.self.value, index)
        reorder_head_matrix(self.output.dense, index, dim=1)

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs # (attention_output, attention_scores, attention_scores_QQ, attention_scores_KK, attention_scores_VV)


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        # dense layer for adaptive width
        self.dense = DynaLinear(config.hidden_size, config.intermediate_size,
                                config.num_attention_heads, dyna_dim=[False, True])
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def reorder_neurons(self, index, dim=0):
        index = index.to(self.dense.weight.device)
        W = self.dense.weight.index_select(dim, index).clone().detach()
        if self.dense.bias is not None:
            if dim == 1:
                b = self.dense.bias.clone().detach()
            else:
                b = self.dense.bias[index].clone().detach()
        self.dense.weight.requires_grad = False
        self.dense.weight.copy_(W.contiguous())
        self.dense.weight.requires_grad = True
        if self.dense.bias is not None:
            self.dense.bias.requires_grad = False
            self.dense.bias.copy_(b.contiguous())
            self.dense.bias.requires_grad = True

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertIntermediate_v1(nn.Module):
    def __init__(self, config):
        super(BertIntermediate_v1, self).__init__()
        # dense layer for adaptive width
        # self.dense = DynaLinear(config.hidden_size, config.intermediate_size, config.num_attention_heads, dyna_dim=[False, True])
        # self.dense = DynaLinear_v1(config.hidden_size, config.intermediate_size, config.num_attention_heads, dyna_dim=[True, True], dyna_cate=['hid', 'hid'])
        self.dense = DynaLinear_v1(config.hidden_size, config.intermediate_size, config.num_attention_heads, dyna_dim=[True, True], dyna_cate=['hid', 'inter'])
        
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def reorder_neurons(self, index, dim=0):
        index = index.to(self.dense.weight.device)
        W = self.dense.weight.index_select(dim, index).clone().detach()
        if self.dense.bias is not None:
            if dim == 1:
                b = self.dense.bias.clone().detach()
            else:
                b = self.dense.bias[index].clone().detach()
        self.dense.weight.requires_grad = False
        self.dense.weight.copy_(W.contiguous())
        self.dense.weight.requires_grad = True
        if self.dense.bias is not None:
            self.dense.bias.requires_grad = False
            self.dense.bias.copy_(b.contiguous())
            self.dense.bias.requires_grad = True

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        # dense layer for adaptive width
        self.dense = DynaLinear(config.intermediate_size, config.hidden_size,
                                  config.num_attention_heads, dyna_dim=[True, False])
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def reorder_neurons(self, index, dim=1):
        index = index.to(self.dense.weight.device)
        W = self.dense.weight.index_select(dim, index).clone().detach()
        if self.dense.bias is not None:
            if dim == 1:
                b = self.dense.bias.clone().detach()
            else:
                b = self.dense.bias[index].clone().detach()
        self.dense.weight.requires_grad = False
        self.dense.weight.copy_(W.contiguous())
        self.dense.weight.requires_grad = True
        if self.dense.bias is not None:
            self.dense.bias.requires_grad = False
            self.dense.bias.copy_(b.contiguous())
            self.dense.bias.requires_grad = True

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertOutput_v1(nn.Module):
    def __init__(self, config):
        super(BertOutput_v1, self).__init__()
        # dense layer for adaptive width
        # self.dense = DynaLinear(config.intermediate_size, config.hidden_size, config.num_attention_heads, dyna_dim=[True, False])
        # self.dense = DynaLinear_v1(config.intermediate_size, config.hidden_size, config.num_attention_heads, dyna_dim=[True, True], dyna_cate=['hid', 'hid'])
        self.dense = DynaLinear_v1(config.intermediate_size, config.hidden_size, config.num_attention_heads, dyna_dim=[True, True], dyna_cate=['inter', 'hid'])

        # self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm = DynaBertLayerNorm_v1(config.hidden_size, eps=config.layer_norm_eps, dyna_dim=[True], dyna_cate=['hid'])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def reorder_neurons(self, index, dim=1):
        index = index.to(self.dense.weight.device)
        W = self.dense.weight.index_select(dim, index).clone().detach()
        if self.dense.bias is not None:
            if dim == 1:
                b = self.dense.bias.clone().detach()
            else:
                b = self.dense.bias[index].clone().detach()
        self.dense.weight.requires_grad = False
        self.dense.weight.copy_(W.contiguous())
        self.dense.weight.requires_grad = True
        if self.dense.bias is not None:
            self.dense.bias.requires_grad = False
            self.dense.bias.copy_(b.contiguous())
            self.dense.bias.requires_grad = True

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.output_intermediate = config.output_intermediate

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        if self.output_intermediate:
            outputs = (layer_output,) + attention_outputs[1:] + (intermediate_output,)
        else:
            outputs = (layer_output,) + attention_outputs[1:]

        return outputs


class BertLayer_v1(nn.Module):
    def __init__(self, config):
        super(BertLayer_v1, self).__init__()
        self.attention = BertAttention_v1(config)
        self.intermediate = BertIntermediate_v1(config)
        self.output = BertOutput_v1(config)
        self.output_intermediate = config.output_intermediate

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        if self.output_intermediate:
            # (layer_output, attention_scores, attention_scores_QQ, attention_scores_KK, attention_scores_VV, intermediate_output)
            outputs = (layer_output,) + attention_outputs[1:] + (intermediate_output,)
        else:
            # (layer_output, attention_scores, attention_scores_QQ, attention_scores_KK, attention_scores_VV)
            outputs = (layer_output,) + attention_outputs[1:]

        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.output_intermediate = config.output_intermediate
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.depth_mult = 1.

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        all_intermediate = ()

        # uniformly remove layers
        depth = round(self.depth_mult * len(self.layer))
        kept_layers_index = []

        # (0,2,4,6,8,10)
        for i in range(depth):
            kept_layers_index.append(math.floor(i/self.depth_mult))

        for i in kept_layers_index:
            layer_module = self.layer[i]
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            if self.output_intermediate:
                all_intermediate = all_intermediate + (layer_outputs[2],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        if self.output_intermediate:
            outputs = outputs + (all_intermediate,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertEncoder_v1(nn.Module):
    def __init__(self, config):
        super(BertEncoder_v1, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.output_intermediate = config.output_intermediate
        self.layer = nn.ModuleList([BertLayer_v1(config) for _ in range(config.num_hidden_layers)])
        self.depth_mult = 1.

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        all_intermediate = ()
        # added by DK
        all_attentions_QQ = ()
        all_attentions_KK = ()
        all_attentions_VV = ()

        # uniformly remove layers
        depth = round(self.depth_mult * len(self.layer))
        kept_layers_index = []

        # (0,2,4,6,8,10)
        # (0,1,2,3,4,5)
        for i in range(depth):
            kept_layers_index.append(math.floor(i/self.depth_mult))
            # kept_layers_index.append(math.floor(i))

        for i in kept_layers_index:
            layer_module = self.layer[i]
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            # if self.output_attentions:
            #     all_attentions = all_attentions + (layer_outputs[1],)
            # if self.output_intermediate:
            #     all_intermediate = all_intermediate + (layer_outputs[2],)

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                all_attentions_QQ = all_attentions_QQ + (layer_outputs[2],)
                all_attentions_KK = all_attentions_KK + (layer_outputs[3],)
                all_attentions_VV = all_attentions_VV + (layer_outputs[4],)
            if self.output_intermediate:
                all_intermediate = all_intermediate + (layer_outputs[5],)


        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
            outputs = outputs + (all_attentions_QQ,)
            outputs = outputs + (all_attentions_KK,)
            outputs = outputs + (all_attentions_VV,)
        if self.output_intermediate:
            outputs = outputs + (all_intermediate,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions), (all attentions QQ), (all attentions KK), (all attentions VV), (all_intermediate)


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPooler_v1(nn.Module):
    def __init__(self, config):
        super(BertPooler_v1, self).__init__()

        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = DynaLinear_v1(config.hidden_size, config.hidden_size, config.num_attention_heads, dyna_dim=[True, True], dyna_cate=['hid', 'hid'])
        
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertPredictionHeadTransform_v1(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform_v1, self).__init__()
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = DynaLinear_v1(config.hidden_size, config.hidden_size, config.num_attention_heads, dyna_dim=[True, True], dyna_cate=['hid', 'hid'])

        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        
        # self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm = DynaBertLayerNorm_v1(config.hidden_size, eps=config.layer_norm_eps, dyna_dim=[True], dyna_cate=['hid'])

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        # extended_attention_mask = extended_attention_mask.to(dtype=torch.float32) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
            # head_mask = head_mask.to(dtype=torch.float32) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs


class BertModel_v1(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel_v1, self).__init__(config)

        self.embeddings = BertEmbeddings_v1(config)
        self.encoder = BertEncoder_v1(config)
        self.pooler = BertPooler_v1(config)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        # extended_attention_mask = extended_attention_mask.to(dtype=torch.float32) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
            # head_mask = head_mask.to(dtype=torch.float32) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # sequence_output, pooled_output, (all hidden states), (all attentions), (all attentions QQ), (all attentions KK), (all attentions VV), (all_intermediate)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs


class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            Attentions scores before softmax.
        **intermediates**: (`optional`, returned when ``config.output_intermediate=True``)
            representation in the intermediate layer after nonlinearity.
    """
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        # print('pooled_output:', pooled_output.shape)
        logits = self.classifier(pooled_output)
        # print('logits:', logits.shape)
        # print('labels:', labels.shape)
        # print('labels:', labels)
        # print('labels.view(-1):', labels.view(-1).shape)
        # print('')

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  regression task
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions), (intermediates)


class BertForSequenceClassification_v1(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification_v1, self).__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel_v1(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.classifier = DynaLinear_v1(config.hidden_size, self.config.num_labels, config.num_attention_heads, dyna_dim=[True, False], dyna_cate=['hid', 'hid'])

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        # outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        # print('pooled_output:', pooled_output.shape)
        logits = self.classifier(pooled_output)
        # print('logits:', logits.shape)
        # print('labels:', labels.shape)
        # print('labels:', labels)
        # print('labels.view(-1):', labels.view(-1).shape)
        # print('')

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  regression task
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions), (intermediates)


class BertForSequenceClassification_v1_AddHidLoss(BertPreTrainedModel):
    def __init__(self, config, fit_size=768):
        super(BertForSequenceClassification_v1_AddHidLoss, self).__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel_v1(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.classifier = DynaLinear_v1(config.hidden_size, self.config.num_labels, config.num_attention_heads, dyna_dim=[True, False], dyna_cate=['hid', 'hid'])
        # self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        self.fit_dense = DynaLinear_v1(config.hidden_size, fit_size, config.num_attention_heads, dyna_dim=[True, False], dyna_cate=['hid', 'hid'])
        self.init_weights()
        

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, is_student=False):

        # outputs = sequence_output, pooled_output, (all hidden states), (all attentions), (all attentions QQ), (all attentions KK), (all attentions VV), (all_intermediate)
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        pooled_output = outputs[1]
        sequence_output = outputs[2] # hidden_states

        pooled_output = self.dropout(pooled_output)
        # print('pooled_output:', pooled_output.shape)
        logits = self.classifier(pooled_output)

        tmp = []
        if is_student:
            for s_id, sequence_layer in enumerate(sequence_output):
                tmp.append(self.fit_dense(sequence_layer))
            sequence_output = tmp

        # outputs = (logits,) + outputs[2:]
        outputs = (logits, sequence_output,) + outputs[3:]

        if labels is not None:
            if self.num_labels == 1:
                #  regression task
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                # print('')
                # print('logits, labels', logits.shape, labels.shape)
                # print('')
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # loss, logits, (sequence_output), (all attentions), (all attentions QQ), (all attentions KK), (all attentions VV), (all_intermediate)


class BertForSequenceClassification_v1_AddHidLoss_ForPreTraining(BertPreTrainedModel):
    def __init__(self, config, fit_size=768):
        super(BertForSequenceClassification_v1_AddHidLoss_ForPreTraining, self).__init__(config)

        self.num_labels = config.num_labels
        self.bert = BertModel_v1(config)

        self.cls = BertPreTrainingHeads_v1(config, self.bert.embeddings.word_embeddings.weight)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.classifier = DynaLinear_v1(config.hidden_size, self.config.num_labels, config.num_attention_heads, dyna_dim=[True, False], dyna_cate=['hid', 'hid'])
        # self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        self.fit_dense = DynaLinear_v1(config.hidden_size, fit_size, config.num_attention_heads, dyna_dim=[True, False], dyna_cate=['hid', 'hid'])
        self.init_weights()
        

    # def forward(self, input_ids, attention_mask=None, token_type_ids=None,
    #             position_ids=None, head_mask=None, labels=None, is_student=False):

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
            position_ids=None, head_mask=None, labels=None, is_student=False, masked_lm_labels=None, next_sentence_label=None):

        # outputs = sequence_output, pooled_output, (all hidden states), (all attentions), (all attentions QQ), (all attentions KK), (all attentions VV), (all_intermediate)
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        pooled_output = outputs[1]
        sequence_output = outputs[2] # hidden_states
        
        # from BertForPreTraining_v1
        if is_student:
            sequence_output_real = outputs[0]
            prediction_scores, seq_relationship_score = self.cls(sequence_output_real, pooled_output)
        else:
            prediction_scores, seq_relationship_score = None, None

        pooled_output = self.dropout(pooled_output)
        # print('pooled_output:', pooled_output.shape)
        logits = self.classifier(pooled_output)

        tmp = []
        if is_student:
            for s_id, sequence_layer in enumerate(sequence_output):
                tmp.append(self.fit_dense(sequence_layer))
            sequence_output = tmp

        # outputs = (logits, sequence_output,) + outputs[3:]
        # outputs = (logits, sequence_output, prediction_scores, seq_relationship_score,) + outputs[3:]
        outputs = (logits, sequence_output,) + outputs[3:]

        # labels == None
        if labels is not None:
            if self.num_labels == 1:
                #  regression task
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                # print('')
                # print('logits, labels', logits.shape, labels.shape)
                # print('')
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        # from BertForPreTraining_v1
        if masked_lm_labels is not None and next_sentence_label is not None:
            if is_student:
                loss_fct = CrossEntropyLoss(ignore_index=-1)
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
                next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
                total_loss = masked_lm_loss + next_sentence_loss
                outputs = (total_loss,) + outputs
            else:
                outputs = (None,) + outputs

        # because labels == None, no hard_loss
        return outputs  # total_loss, logits, (sequence_output), (all attentions), (all attentions QQ), (all attentions KK), (all attentions VV), (all_intermediate)


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertLMPredictionHead_v1(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead_v1, self).__init__()
        self.transform = BertPredictionHeadTransform_v1(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        # self.decoder = nn.Linear(config.hidden_size,
        #                          config.vocab_size,
        #                          bias=False)

        # bert_model_embedding_weights = [hidden, vocab]
        self.decoder = DynaLinear_v1(config.hidden_size, config.vocab_size, config.num_attention_heads, dyna_dim=[True, False], dyna_cate=['hid', 'hid'], bias=False)
        self.decoder.weight = bert_model_embedding_weights

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainingHeads_v1(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads_v1, self).__init__()
        # self.predictions = BertLMPredictionHead(config)
        self.predictions = BertLMPredictionHead_v1(config, bert_model_embedding_weights)
        # self.seq_relationship = nn.Linear(config.hidden_size, 2)
        self.seq_relationship_v1 = DynaLinear_v1(config.hidden_size, 2, config.num_attention_heads, dyna_dim=[True, False], dyna_cate=['hid', 'hid'])

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        # seq_relationship_score = self.seq_relationship(pooled_output)
        seq_relationship_score = self.seq_relationship_v1(pooled_output)
        return prediction_scores, seq_relationship_score


class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                masked_lm_labels=None, next_sentence_label=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


class BertForPreTraining_v1(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForPreTraining_v1, self).__init__(config)

        self.bert = BertModel_v1(config)
        # self.cls = BertPreTrainingHeads(config)
        # self.cls = BertPreTrainingHeads_v1(config, self.bert.embeddings.word_embeddings.embedding.weight)
        self.cls = BertPreTrainingHeads_v1(config, self.bert.embeddings.word_embeddings.weight)

        self.init_weights()
        # self.tie_weights()

    # def tie_weights(self):
    #     """ Make sure we are sharing the input and output embeddings.
    #         Export to TorchScript can't handle parameter sharing so we are cloning them instead.
    #     """
    #     self._tie_or_clone_weights(self.cls.predictions.decoder,
    #                                self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                masked_lm_labels=None, next_sentence_label=None):

        # outputs = sequence_output, pooled_output, (all hidden states), (all attentions), (all attentions QQ), (all attentions KK), (all attentions VV), (all_intermediate)
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]  # add others in "outputs"

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, () ... ()