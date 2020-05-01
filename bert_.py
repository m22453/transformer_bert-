# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# update by 22453
"""PyTorch BERT model+. """

import logging
import math
import os

import torch
from torch import nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss, MSELoss


from transformers import (
    BertPreTrainedModel,
    BertTokenizer,
    BertConfig,
    BertModel,
)


logger = logging.getLogger(__name__)


class BertForSequenceClassification_rnn(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # add lstm params
        self.rnn_hidden_ = 768
        self.num_layers_ = 2
        self.dropout_ = 0.1

        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.lstm = nn.LSTM(config.hidden_size, self.rnn_hidden_, self.num_layers_,
                            bidirectional=True, batch_first=True, dropout=self.dropout_)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.rnn_hidden_*2+config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # get the sequence output
        sequence_output = outputs[0]
        lstm_out, _ = self.lstm(sequence_output)

        pooled_output = self.dropout(lstm_out)

        # hidden state of the sequence end by fc
        logits = self.classifier(pooled_output[:, -1, :])

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForSequenceClassification_cnn(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # add cnn params
        self.filter_sizes_ = (3, 4, 5)  # 卷积核尺寸
        self.num_filters_ = 256  # 卷积核数量(channels数)
        self.hidden_size_ = 768

        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters_, (k, self.hidden_size_)) for k in self.filter_sizes_])

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.num_filters_ * len(self.filter_sizes_)+self.hidden_size_, config.num_labels)

        self.init_weights()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # get the sequence output
        sequence_output = outputs[0]
        out = sequence_output.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)

        out = torch.cat((out,outputs[1]),1)
        pooled_output = self.dropout(out)

        # by fc
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForSequenceClassification_rcnn(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # add dpcnn params
        self.filter_sizes_ = (3, 4, 5)  # 卷积核尺寸
        self.num_filters_ = 250  # 卷积核数量(channels数)
        self.num_layers_ = 2
        self.dropout_ = 0.1
        self.rnn_hidden_ = 768
        self.max_seq_length = 150 #与模型输入保持一致


        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        self.lstm = nn.LSTM(config.hidden_size, self.rnn_hidden_, self.num_layers_,
                            bidirectional=True, batch_first=True, dropout=self.dropout_)
        self.maxpool = nn.MaxPool1d(self.max_seq_length)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.rnn_hidden_ * 2 + config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # get the sequence output
        sequence_output = outputs[0]
        lstm_out, _ = self.lstm(sequence_output)

        co_out = torch.cat((sequence_output,lstm_out),2)
        co_out = F.relu(co_out)
        co_out = co_out.permute(0,2,1)
        co_out = self.maxpool(co_out).squeeze()
        #co_out = torch.cat((co_out,outputs[1]),1)
        co_out = self.dropout(co_out)

        logits = self.classifier(co_out)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)



class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import BertTokenizer, BertForTokenClassification
        import torch
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
