import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleBert(nn.Module):
    def __init__(self, seq_len, output_size):
        super(SimpleBert, self).__init__()
        self.seq_len = seq_len
        self.output_size = output_size
        self.d_model = 768
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # 12-layer, 768-hidden, 12-heads, 110M parameters
        # output: [batch_size, sequence_length, hidden_size]
        # choose the hidden of first token [CLS]
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
        bert_output = bert_feature[11]
        bert_output = bert_output[:, 0, :]
        context = self.linear(bert_output)
        outputs = self.softmax(context)
        return outputs


class RecBert(nn.Module):
    def __init__(self, seq_len, hidden_size, output_size, bidirec=True):
        super(RecBert, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.d_model = 768
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # 12-layer, 768-hidden, 12-heads, 110M parameters
        # output: [batch_size, seq_length, d_model]
        # self.linear = nn.Linear(self.d_model * self.seq_len, self.output_size)
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=hidden_size, batch_first=True, bidirectional=bidirec,
                            dropout=0.5, num_layers=2)
        # hidden: [4, N, hidden]
        # choose the hidden of last layer
        self.linear = nn.Linear(self.hidden_size * 2, self.output_size)
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
        bert_output = bert_feature[11]
        context, (hidden, cell) = self.lstm(bert_output)  # N * seq_len * hidden_size
        hidden = torch.cat([hidden[-1], hidden[-2]], dim=-1)  # hidden state of the last layer [N, hidden*2]
        outputs = self.softmax(self.linear(hidden))  # N * output_size
        return outputs
