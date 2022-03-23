import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import nltk
import nltk.tokenize as tokenizer
from nltk.corpus import stopwords

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
uncased_bert_vocab_size = 30522
uncased_bert_emb_size = 768


class TextRNN(nn.Module):
    def __init__(self, seq_len, hidden_size, output_size, vocab_size=uncased_bert_vocab_size,
                 emb_size=uncased_bert_emb_size, bidirec=True):
        super(TextRNN, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.bidirec = bidirec
        self.embedding = nn.Embedding(vocab_size, emb_size)  # N * seq_len * emb_size
        self.lstm = nn.LSTM(input_size=self.emb_size, hidden_size=hidden_size, batch_first=True, bidirectional=bidirec,
                            dropout=0.5, num_layers=2)
        # hidden: 2D, N, hidden
        self.linear = nn.Linear(self.hidden_size * (2 if bidirec else 1), self.output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, length):
        '''
        :param length: N (length of valid seq)
        :param inputs: N * seq_len (tokenized and indexed)
        -> embedding: N * seq_len * emb_size
        -> lstm: N * seq_len * hidden_size
        -> linear: N * output_size
        :return: N * output_size
        '''
        emb = self.embedding(inputs)
        emb = pack_padded_sequence(emb, length.to("cpu"), batch_first=True, enforce_sorted=False)
        context, (hidden, cell) = self.lstm(emb)
        hidden = torch.cat([hidden[-1], hidden[-2]], dim=-1) if self.bidirec else hidden[-1]
        #  select the final hidden state of the last layer [N, hidden*D]
        outputs = self.softmax(self.linear(hidden))  # N * output_size
        return outputs


class TextCNN(nn.Module):
    def __init__(self, seq_len, hidden_size, output_size, kernel_size, vocab_size=uncased_bert_vocab_size,
                emb_size=uncased_bert_emb_size):
        super(TextCNN, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.kernel_size = kernel_size  # tuple
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.embedding = nn.Embedding(vocab_size, emb_size)  # N * seq_len * emb_size
        # stride(1), padding(0), dilation(1) are set to default
        # input: [N, 1, seq_len, emb_size]
        height1 = (seq_len - kernel_size[0] + 1) // 2
        width1 = (emb_size - kernel_size[1] + 1) // 2
        height2 = (height1 - kernel_size[0] + 1) // 2
        width2 = (width1 - kernel_size[1] + 1) // 2
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, hidden_size, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # [N, hidden_size, height, width]
        self.cnn2 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # [N, hidden_size * 2, height2, width2]
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(hidden_size * 2 * height2 * width2, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        emb = self.embedding(inputs)  # N * seq_len * emb_size
        emb = emb.unsqueeze(dim=1)
        context = self.cnn1(emb)
        context = self.cnn2(context)
        context = self.dropout(context)
        batch_size = context.shape[0]
        context = context.view(batch_size, -1)
        output = self.softmax(self.linear(context))
        return output


class BasicTokenizer:
    """
    We won't use this any more.
    """
    def __init__(self, sentence_lst, language='english'):
        self.stopwords = set(stopwords.words(language))
        self.word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
        self.token_lst = []
        self.index_lst = []
        self.vocab_size = 4
        self.tokenize(sentence_lst, language)

    def tokenize(self, sentence_lst, language='english'):
        for sentence in sentence_lst:
            words = tokenizer.word_tokenize(sentence.lower(), language)
            tokens = ['[CLS]']
            index = [self.word2idx['[CLS]']]
            for i in words:
                if i not in self.stopwords:
                    tokens.append(i)
                    index.append(self.word2idx[i])
            tokens.append('[SEP]')
            index.append(self.word2idx['[SEP]'])
            self.token_lst.append(tokens)
            self.index_lst.append(index)
