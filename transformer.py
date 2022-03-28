import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import torch.utils.data as Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def getPadMask(seq_q, seq_k, pad):
    """
        :param seq_q: N * len_q
        :param seq_k: N * len_k
        :param pad: the index of [PAD]
        :return: mask: N * len_q * len_k
        """
    batch_size, len_q = seq_q.size()
    _, len_k = seq_k.size()
    mask = torch.zeros([batch_size, len_q, len_k]).to(device)
    matrix_q = seq_q.unsqueeze(1).transpose(-1, -2).repeat(1, 1, len_k).to(device)
    matrix_k = seq_k.unsqueeze(1).repeat(1, len_q, 1).to(device)
    mask[matrix_q == pad] = 1
    mask[matrix_k == pad] = 1
    return mask


def getDecoderMask(seq):
    """
    :param seq: tgt sequence: N * tgt_len
    :return: mask: N * tgt_len * tgt_len
    """
    batch_size, tgt_len = seq.size()
    mask = torch.zeros(batch_size, tgt_len, tgt_len).to(device)
    row_idx = torch.arange(0, tgt_len, 1).unsqueeze(0).unsqueeze(0).transpose(-1, -2).repeat(batch_size, 1, tgt_len).to(device)
    col_idx = torch.arange(0, tgt_len, 1).unsqueeze(0).unsqueeze(0).repeat(batch_size, tgt_len, 1).to(device)
    mask[row_idx < col_idx] = 1
    mask[row_idx >= col_idx] = 0
    return mask


def getPositionalEmbedding(batch_size, seq_len, d_model):
    """
        :param seq_len:  length of input sequence
        :param d_model:  dimension of embedding
        :return: pos_emb : N * seq_len * d_model
        """
    pos_idx = torch.arange(0, seq_len).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, d_model).to(device)
    pow_idx = torch.arange(0, d_model).to(device)
    pow_idx[pow_idx % 2 == 1] -= 1
    pow_idx = pow_idx.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1)
    pos_emb = pos_idx / torch.pow(10000, pow_idx / d_model).to(device)
    pos_emb[:, :, 0::2] = torch.sin(pos_emb[:, :, 0::2])
    pos_emb[:, :, 1::2] = torch.cos(pos_emb[:, :, 1::2])
    return pos_emb


class Configuration:
    def __init__(self):
        self.d_model = 512
        self.src_len = 512
        self.tgt_len = 512
        self.src_vocab_size = 30522
        self.tgt_vocab_size = 30522
        self.dimq = 64
        self.dimv = 64
        self.n_heads = 8
        self.d_hidden = 2048
        self.n_layer = 6
        self.code_dict = {"pad": 0}


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, dimq, dimv, n_heads):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.dimq = dimq
        self.dimk = dimq
        self.dimv = dimv
        self.WQ = nn.Linear(in_features=d_model, out_features=dimq * n_heads, bias=False)
        self.WK = nn.Linear(in_features=d_model, out_features=dimq * n_heads, bias=False)
        self.WV = nn.Linear(in_features=d_model, out_features=dimv * n_heads, bias=False)
        self.linear = nn.Linear(in_features=dimv * n_heads, out_features=d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, inputQ, inputK, inputV, mask):
        """
        :param inputQ: N * seq_len * d_model (seq_len could be src_len or tgt_len)
        :param inputK: N * seq_len * d_model
        :param inputV: N * seq_len * d_model
        :param mask: N * seq_len * seq_len
        :return: output: N * seq_len * d_model
        """
        Q = self.WQ(inputQ)  # N * seq_len * (n_heads*dim_q)
        K = self.WK(inputK)  # N * seq_len * (n_heads*dim_k)
        V = self.WV(inputV)  # N * seq_len * (n_heads*dim_v)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(
            torch.tensor([self.dimk]).to(device))
        scores = scores.to(device) # N * seq_len * seq_len
        scores[mask == 1] = -1e9
        weights = torch.matmul(F.softmax(scores, dim=-1), V)  # N * seq_len * (n_heads*dim_v)
        weights = weights.to(device)
        output = self.layernorm(self.linear(weights) + inputQ)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=d_hidden, out_features=d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """
        :param inputs: N * seq_len * d_model (seq_len could be src_len or tgt_len)
        :return: output: N * seq_len * d_model
        """
        context = self.linear1(inputs)
        context = self.relu(context)
        context = self.linear2(context)
        output = self.layernorm(inputs + context)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, conf):
        super(EncoderLayer, self).__init__()
        self.multiheadSelfAttention = MultiheadAttention(conf.d_model, conf.dimq, conf.dimv, conf.n_heads)
        self.feedForward = FeedForward(conf.d_model, conf.d_hidden)

    def forward(self, inputs, mask):
        """
        :param inputs: N * src_len * d_model
        :param mask: N * src_len * src_len
        :return: output: N * src_len * d_model
        """
        attn = self.multiheadSelfAttention(inputs, inputs, inputs, mask)
        output = self.feedForward(attn)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, conf):
        super(TransformerEncoder, self).__init__()
        self.conf = conf
        self.word_emb = nn.Embedding(conf.src_vocab_size, conf.d_model)
        self.layers = nn.ModuleList([EncoderLayer(conf) for _ in range(conf.n_layer)])

    def forward(self, input_seq, pad_mask):
        """
        :param input_seq:  N * src_len
        :return: N * src_len * d_model
        """
        batch_size = input_seq.shape[0]
        wemb = self.word_emb(input_seq)
        pemb = getPositionalEmbedding(batch_size, self.conf.src_len, self.conf.d_model).to(device)
        encoder_output = wemb + pemb
        # encoder_multiselfattn_mask = getPadMask(input_seq, input_seq, self.conf.code_dict["pad"])
        for layer in self.layers:
            encoder_output = layer(encoder_output, pad_mask)

        return encoder_output


class DecoderLayer(nn.Module):
    def __init__(self, conf):
        super(DecoderLayer, self).__init__()
        self.conf = conf
        self.multiheadSelfAttention = MultiheadAttention(conf.d_model, conf.dimq, conf.dimv, conf.n_heads)
        self.multiheadAttention = MultiheadAttention(conf.d_model, conf.dimq, conf.dimv, conf.n_heads)
        self.feedForward = FeedForward(conf.d_model, conf.d_hidden)

    def forward(self, inputs, encoder_output, self_mask, cross_mask):
        """
        :param encoder_output: N * src_len * d_model
        :param inputs:  N * tgt_len * d_model
        :param self_mask:  N * tgt_len * tgt_len
        :param cross_mask:  N * tgt_len * seq_len
        :return:  N * tgt_len * d_model
        """
        attn1 = self.multiheadSelfAttention(inputs, inputs, inputs, self_mask)
        attn2 = self.multiheadAttention(attn1, encoder_output, encoder_output, cross_mask)
        output = self.feedForward(attn2)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, conf):
        super(TransformerDecoder, self).__init__()
        self.conf = conf
        self.word_emb = nn.Embedding(conf.tgt_vocab_size, conf.d_model)
        self.layers = nn.ModuleList([DecoderLayer(conf) for _ in range(conf.n_layer)])

    def forward(self, input_seq, encoder_output, self_mask, cross_mask):
        """
        :param input_seq: N * tgt_len
        :param encoder_output:  N * src_len * d_model
        :return:  N * tgt_len * d_model
        """
        batch_size = input_seq.shape[0]
        wemb = self.word_emb(input_seq)
        pemb = getPositionalEmbedding(batch_size, self.conf.tgt_len, self.conf.d_model).to(device)
        # pad_mask = getPadMask(input_seq, input_seq, self.conf.code_dict["pad"])
        # dec_mask = getDecoderMask(input_seq)
        # cross_mask = getPadMask(input_seq, encoder_output, self.conf.code_dict["pad"])
        decoder_output = wemb + pemb
        for layer in self.layers:
            decoder_output = layer(decoder_output, encoder_output, self_mask, cross_mask)

        return decoder_output


class Transformer(nn.Module):
    def __init__(self, conf):
        super(Transformer, self).__init__()
        self.conf = conf
        self.encoder = TransformerEncoder(conf)
        self.decoder = TransformerDecoder(conf)
        self.linear = nn.Linear(conf.d_model, conf.tgt_vocab_size)

    def forward(self, encoder_input, decoder_input):
        """
        :param encoder_input: N * src_len
        :param decoder_input: N * tgt_len
        :return: N * tgt_len
        """
        encoder_mask = getPadMask(encoder_input, encoder_input, self.conf.code_dict["pad"]).to(device)
        decoder_mask = (getPadMask(decoder_input, decoder_input, self.conf.code_dict["pad"]) + getDecoderMask(
            decoder_input)).to(device)
        cross_mask = getPadMask(decoder_input, encoder_input, self.conf.code_dict["pad"]).to(device)
        encoder_output = self.encoder(encoder_input, encoder_mask)
        decoder_output = self.decoder(decoder_input, encoder_output, decoder_mask, cross_mask)
        prob_matrix = F.softmax(self.linear(decoder_output), dim=-1)  # N * tgt_len * tgt_vocab_size
        output_seq = torch.argmax(prob_matrix, dim=-1)
        return output_seq, prob_matrix


'''
if __name__ == "__main__":
    conf = Configuration()

    model = Transformer(conf).cuda()

    print(model)

    src = torch.tensor([[3, 4, 5, 6, 7], [3, 5, 4, 7, 0]]).cuda()

    tgt = torch.tensor([[4, 2, 5, 3, 0, 0], [9, 5, 7, 4, 3, 5]]).cuda()

    output = model(src, tgt)
    print(src.size(), src.dtype)
    print(tgt.size(), tgt.dtype)
    print(output.size(), output.dtype)
'''
if __name__ == "__main__":

    conf = Configuration()

    sentences = [
        # enc_input           dec_input         dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
    ]

    # Padding Should be Zero
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
    conf.src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
    idx2word = {i: w for i, w in enumerate(tgt_vocab)}
    conf.tgt_vocab_size = len(tgt_vocab)

    conf.src_len = 5  # enc_input max sequence length
    conf.tgt_len = 6  # dec_input(=dec_output) max sequence length


    def make_data(sentences):
        enc_inputs, dec_inputs, dec_outputs = [], [], []
        for i in range(len(sentences)):
            enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]  # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
            dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]  # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
            dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]  # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]

            enc_inputs.extend(enc_input)
            dec_inputs.extend(dec_input)
            dec_outputs.extend(dec_output)

        return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


    enc_inputs, dec_inputs, dec_outputs = make_data(sentences)


    class MyDataSet(Data.Dataset):
        def __init__(self, enc_inputs, dec_inputs, dec_outputs):
            super(MyDataSet, self).__init__()
            self.enc_inputs = enc_inputs
            self.dec_inputs = dec_inputs
            self.dec_outputs = dec_outputs

        def __len__(self):
            return self.enc_inputs.shape[0]

        def __getitem__(self, idx):
            return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

    conf.code_dict["start"] = 6
    conf.code_dict["end"] = 7

    model = Transformer(conf).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=conf.code_dict["pad"])
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.99)
    for epoch in range(30):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            '''
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            '''
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs, prob_matrix = model(enc_inputs, dec_inputs)
            loss = criterion(prob_matrix.view(prob_matrix.shape[0] * prob_matrix.shape[1], prob_matrix.shape[2]),
                             dec_outputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
