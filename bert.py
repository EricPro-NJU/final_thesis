import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_pretrained_bert import BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleBert(nn.Module):
    def __init__(self, seq_len, output_size, language="english", feat=0):
        super(SimpleBert, self).__init__()
        self.seq_len = seq_len
        self.output_size = output_size
        self.d_model = 768
        self.bert = BertModel.from_pretrained('bert-base-uncased' if language == "english" else 'bert-base-chinese')
        self.feat = feat
        # 12-layer, 768-hidden, 12-heads, 110M parameters
        # output: [batch_size, sequence_length, hidden_size]
        # choose the hidden of first token [CLS]
        if feat == 2:
            self.params = nn.ParameterList(
                [nn.Parameter(torch.tensor([1 / 12], device=device), requires_grad=True) for _ in range(12)])
        self.linear = nn.Linear(self.d_model, self.output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, mask):
        '''
        :param inputs: N * seq_len
        :param mask: N * seq_len
        :var bert_output: N * seq_len * hidden_size
        :return: N * output_size (after softmax, represent probability)
            classification logits
        '''
        bert_feature, _ = self.bert(inputs, attention_mask=mask)
        if self.feat == 0:
            bert_output = bert_feature[11]
            bert_output = bert_output[:, 0, :]
        elif self.feat == 1:
            temp = torch.cat([item[:, 0, :].unsqueeze(0) for item in bert_feature], dim=0)
            bert_output = torch.mean(temp, dim=0)
        elif self.feat == 2:
            temp = torch.cat([item[:, 0, :].unsqueeze(0) for item in bert_feature], dim=0)  # 12, N, hidden
            bert_output = self.params[0] * temp[0] + self.params[1] * temp[1] + self.params[2] * temp[2] + self.params[
                3] * temp[3] + self.params[4] * temp[4] + self.params[5] * temp[5] + self.params[6] * temp[6] + self.params[
                7] * temp[7] + self.params[8] * temp[8] + self.params[9] * temp[9] + self.params[10] * temp[10] + self.params[
                11] * temp[11]
        else:
            raise ValueError
        context = self.linear(bert_output)
        outputs = self.softmax(context)
        return outputs


class RecBert(nn.Module):
    def __init__(self, seq_len, hidden_size, output_size, bidirec=True, language="english", method=0, num_layers=2):
        super(RecBert, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.d_model = 768
        self.bidirec = bidirec
        self.method = method
        self.num_layers = num_layers
        self.bert = BertModel.from_pretrained('bert-base-uncased' if language == "english" else 'bert-base-chinese')
        # 12-layer, 768-hidden, 12-heads, 110M parameters
        # bert_output: [batch_size, seq_length, d_model]
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=hidden_size, batch_first=True, bidirectional=bidirec,
                            dropout=0.5, num_layers=num_layers)
        # self.dropout = nn.Dropout(p=0.5)
        # hidden: [D * num_of_layers, N, hidden]
        # choose the hidden of last layer
        input_idx = [(2 if bidirec else 1), (2 if bidirec else 1), (2 if bidirec else 1) * self.seq_len, 1]
        self.linear = nn.Linear(self.hidden_size * input_idx[method], self.output_size)
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, mask, length):
        '''
        :param inputs: N * seq_len
        :param mask: N * seq_len
        :var bert_output: N * seq_len * hidden_size
        :return: N * output_size (after softmax, represent probability)
            classification logits
        '''
        bert_feature, _ = self.bert(inputs, attention_mask=mask)
        bert_output = bert_feature[11]  # N * seq_len * hidden_size
        context, (hidden, cell) = self.lstm(bert_output)
        # _, idx_sort = torch.sort(length, dim=0, descending=True)
        # _, idx_unsort = torch.sort(idx_sort, dim=0)
        # bert_output = bert_output[idx_sort]
        # length = length[idx_sort]
        # bert_output_packed = pack_padded_sequence(input=bert_output, lengths=length.to("cpu"), batch_first=True)
        # packed_context, (hidden, cell) = self.lstm(bert_output_packed)
        # context, context_length = pad_packed_sequence(packed_context, batch_first=True)
        # context = context[idx_unsort]
        # hidden : [(2/1) * num_layers, N, hidden]
        # context: [N, seq_len, (2/1)*hidden]
        if self.method == 0:
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=-1) if self.bidirec else hidden[-1]
            #  select the final hidden state of the last layer [N, hidden*D]
            outputs = self.softmax(self.linear(hidden))  # N * output_size
        elif self.method == 1:
            context = torch.sum(context, dim=1)  # N, (2/1)*hidden
            outputs = self.softmax(self.linear(context))  # N * output_size
        elif self.method == 2:
            context = context.contiguous().view(context.shape[0], -1)
            # select all hidden state: [N, hidden*D*L]
            outputs = self.softmax(self.linear(context))  # N * output_size
        elif self.method == 3:
            hidden = (hidden[-1] + hidden[-2]) if self.bidirec else hidden[-1]
            context = bert_output[:, 0, :] + hidden
            context = self.layernorm(context)
            outputs = self.softmax(self.linear(context))  # N * output_size
        else:
            raise ValueError("Error in RecBert, invalid method")
        return outputs


'''
class RecBertCat(nn.Module):
    def __init__(self, seq_len, hidden_size, output_size, bidirec=True, language="english", num_layers=1):
        super(RecBertCat, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.d_model = 768
        self.vocab_size = 30522
        self.bidirec = bidirec
        self.num_layers = num_layers
        self.bert = BertModel.from_pretrained('bert-base-uncased' if language == "english" else 'bert-base-chinese')
        # 12-layer, 768-hidden, 12-heads, 110M parameters
        # bert_output: [batch_size, seq_length, d_model]
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)  # N * seq_len * emb_size
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=hidden_size, batch_first=True, bidirectional=bidirec,
                            dropout=0.5, num_layers=num_layers)
'''
