import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
uncased_bert_vocab_size = 30522
uncased_bert_emb_size = 768


class SimpleLSTM(nn.Module):
    def __init__(self, seq_len, hidden_size, output_size, vocab_size=uncased_bert_vocab_size,
                 emb_size=uncased_bert_emb_size, bidirec=True):
        super(SimpleLSTM, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size * 2 if bidirec else hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(input_size=self.emb_size, hidden_size=hidden_size, batch_first=True, bidirectional=bidirec,
                            dropout=0.1)
        # [batch_size, seq_len, hidden_size]
        self.layernorm = nn.LayerNorm(self.hidden_size * self.seq_len)
        self.linear = nn.Linear(self.hidden_size * self.seq_len, self.output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, mask=torch.randn([1,1,1])):
        '''
        :param inputs: N * seq_len
        -> embedding: N * seq_len * emb_size
        -> lstm: N * seq_len * hidden_size
        -> linear: N * output_size
        :return: N * output_size
        '''
        emb = self.embedding(inputs)
        context, _ = self.lstm(emb)
        batch_size = context.shape[0]
        context = context.reshape(batch_size, self.seq_len * self.hidden_size)
        outputs = self.softmax(self.linear(self.layernorm(context)))  # N * output_size
        return outputs
