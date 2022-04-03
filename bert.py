import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleBert(nn.Module):
    def __init__(self, seq_len, output_size, language="english"):
        super(SimpleBert, self).__init__()
        self.seq_len = seq_len
        self.output_size = output_size
        self.d_model = 768
        self.bert = BertModel.from_pretrained('bert-base-uncased' if language == "english" else 'bert-base-chinese')
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
    def __init__(self, seq_len, hidden_size, output_size, bidirec=True, language="english"):
        super(RecBert, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.d_model = 768
        self.bidirec = bidirec
        self.bert = BertModel.from_pretrained('bert-base-uncased' if language == "english" else 'bert-base-chinese')
        # 12-layer, 768-hidden, 12-heads, 110M parameters
        # bert_output: [batch_size, seq_length, d_model]
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=hidden_size, batch_first=True, bidirectional=bidirec)
        self.dropout = nn.Dropout(p=0.5)
        # hidden: [D * num_of_layers, N, hidden]
        # choose the hidden of last layer
        self.linear = nn.Linear(self.hidden_size * (2 if bidirec else 1), self.output_size)
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
        # hidden : [(2/1), N, hidden]
        hidden = torch.cat([hidden[-1], hidden[-2]], dim=-1) if self.bidirec else hidden[-1]
        #  select the final hidden state of the last layer [N, hidden*D]
        outputs = self.softmax(self.linear(self.dropout(hidden)))  # N * output_size
        return outputs


class RecBert2(nn.Module):
    def __init__(self, seq_len, hidden_size, output_size, bidirec=True, language="english"):
        super(RecBert2, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.d_model = 768
        self.bidirec = bidirec
        self.bert = BertModel.from_pretrained('bert-base-uncased' if language == "english" else 'bert-base-chinese')
        # 12-layer, 768-hidden, 12-heads, 110M parameters
        # bert_output: [batch_size, seq_length, d_model]
        self.rnn_forward = nn.ModuleList([nn.LSTMCell(input_size=self.d_model, hidden_size=hidden_size)
                                          for _ in range(seq_len)])
        if bidirec:
            self.rnn_backward = nn.ModuleList([nn.LSTMCell(input_size=self.d_model, hidden_size=hidden_size)
                                               for _ in range(seq_len)])
        self.dropout = nn.Dropout(p=0.5)
        # hidden: [2D, N, hidden]
        # choose the hidden of last layer
        self.linear = nn.Linear(self.hidden_size * self.seq_len * (2 if bidirec else 1), self.output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, mask):
        '''
        :param inputs: N * seq_len
        :param mask: N * seq_len
        :var bert_output: N * seq_len * d_model
        :return: N * output_size (after softmax, represent probability)
            classification logits
        '''
        bert_feature, _ = self.bert(inputs, attention_mask=mask)
        bert_output = bert_feature[11]  # N * seq_len * d_model
        batch_size = bert_output.shape[0]
        hidden = torch.zeros([batch_size, self.hidden_size]).to(device)
        cell = torch.zeros([batch_size, self.hidden_size]).to(device)
        feature = torch.zeros([self.seq_len * (2 if self.bidirec else 1), batch_size, self.hidden_size]).to(device)
        for i, layer in enumerate(self.rnn_forward):
            context = bert_output[:, i, :]  # N * d_model
            (hidden, cell) = layer(context, (hidden, cell))
            feature[i] = hidden
        if self.bidirec:
            hidden = torch.zeros([batch_size, self.hidden_size]).to(device)
            cell = torch.zeros([batch_size, self.hidden_size]).to(device)
            for i, layer in enumerate(self.rnn_backward):
                j = self.seq_len - 1 - i
                context = bert_output[:, j, :]
                (hidden, cell) = layer(context, (hidden, cell))
                feature[self.seq_len + i] = hidden
        feature = feature.transpose(0, 1)
        feature = feature.view(batch_size, -1)
        feature = self.dropout(feature)
        output = self.softmax(self.linear(feature))
        return output
