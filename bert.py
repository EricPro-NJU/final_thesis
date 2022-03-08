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
        self.linear = nn.Linear(self.d_model * self.seq_len, self.output_size)
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
        batch_size = bert_feature[11].shape[0]
        context = self.linear(bert_output.view(batch_size, -1))
        outputs = self.softmax(context)
        return outputs
