import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam

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
        print(bert_output.shape)
        batch_size = bert_feature[11].shape[0]
        context = self.linear(bert_output.view(batch_size, -1))
        outputs = self.softmax(context)
        return outputs


def index_data(data_path, data_token_path=None, data_index_path=None, data_mask_path=None, data_label_path=None):
    '''
    Read data from file, and save tokens, indexes, mask if necessary
    :param data_path:
    :param data_token_path: if None, do not save to file
    :param data_index_path:
    :param data_mask_path:
    :param data_label_path:
    :return: 4 lists including data information
    '''

    data_list = []
    token_list = []
    index_list = []
    mask_list = []
    label_list = []

    with open(data_path, "r", encoding="UTF-8-sig") as train_file:
        linereader = train_file.readlines()
        for line in linereader:
            num = eval(line[0])
            sentence = line[2:]
            data_list.append([num, sentence.strip()])

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    for item in data_list:
        tokens = tokenizer.tokenize(item[1])
        tokens_len = len(tokens)
        if tokens_len > 510:
            del tokens[128:-382]
            tokens_len = 510
        tokens.insert(0, "[CLS]")
        tokens.append("[SEP]")
        tokens_len += 2
        pad_num = 512 - tokens_len
        tokens.extend(["[PAD]"] * pad_num)
        mask = [1] * tokens_len + [0] * pad_num
        index = tokenizer.convert_tokens_to_ids(tokens)
        token_list.append(tokens)
        index_list.append(index)
        mask_list.append(mask)
        label_list.append(item[0])

    if data_token_path is not None:
        with open(data_token_path, "w", encoding="UTF-8") as fp:
            for item in token_list:
                fp.write("{}\n".format(item))

    if data_index_path is not None:
        with open(data_index_path, "w", encoding="UTF-8") as fp:
            for item in index_list:
                fp.write("{}\n".format(item))

    if data_mask_path is not None:
        with open(data_mask_path, "w", encoding="UTF-8") as fp:
            for item in mask_list:
                fp.write("{}\n".format(item))

    if data_label_path is not None:
        with open(data_label_path, "w", encoding="UTF-8") as fp:
            for item in label_list:
                fp.write("{}\n".format(item))

    return token_list, index_list, mask_list, label_list


def label_logits(labels, group_num):
    logits = []
    for item in labels:
        prob = [0] * group_num
        prob[item] = 1
        logits.append(prob)
    return logits


class IMDBDataSet(Dataset):
    def __init__(self, src_file, token_file=None, index_file=None, mask_file=None, label_file=None):
        super(IMDBDataSet, self).__init__()
        if src_file is not None:
            token_list, index_list, mask_list, label_list = index_data(src_file, token_file, index_file, mask_file,
                                                                       label_file)
            self.input_idx = torch.LongTensor(index_list)  # num * seq_len
            self.mask_idx = torch.LongTensor(mask_list)  # num * seq_len
            self.label_idx = torch.LongTensor(label_logits(label_list, 2))  # num * 2
        else:
            if index_file is None or mask_file is None or label_file is None:
                raise FileNotFoundError("You should identify source file of data.")
            index = []
            mask = []
            label = []
            with open(index_file, "r", encoding="UTF-8") as fp:
                linereader = fp.readlines()
                for line in linereader:
                    index.append(eval(line))
            with open(mask_file, "r", encoding="UTF-8") as fp:
                linereader = fp.readlines()
                for line in linereader:
                    mask.append(eval(line))
            with open(label_file, "r", encoding="UTF-8") as fp:
                linereader = fp.readlines()
                for line in linereader:
                    label.append(eval(line))
            self.input_idx = torch.LongTensor(index)
            self.mask_idx = torch.LongTensor(mask)
            self.label_idx = torch.LongTensor(label_logits(label, 2))

    def __len__(self):
        return self.input_idx.shape[0]

    def __getitem__(self, idx):
        return self.input_idx[idx], self.mask_idx[idx], self.label_idx[idx]


def fine_tuning_IMDB(task_name, state_path=None):
    # load training data and indexing texts
    print("Indexing Training Data......")
    train_file = "/content/drive/MyDrive/bert/IMDBtrain.csv"
    train_token_file = "/content/drive/MyDrive/bert/IMDBtrain_token.txt"
    train_index_file = "/content/drive/MyDrive/bert/IMDBtrain_index.txt"
    train_mask_file = "/content/drive/MyDrive/bert/IMDBtrain_mask.txt"
    train_label_file = "/content/drive/MyDrive/bert/IMDBtrain_label.txt"
    trainloader = DataLoader(IMDBDataSet(None, train_token_file, train_index_file, train_mask_file, train_label_file),
                             batch_size=32, shuffle=True)
    t_batch = len(trainloader)
    print("Index Training Data Done.")

    # prepare BERT model and set hyper params
    print("Model Config......")
    model = SimpleBert(512, 2).to(device)
    model.train()

    init_epoch = 0
    t_epoch = 4
    lr = 2e-5
    warmup = 0.1
    t_total = 1e5

    criterion = nn.CrossEntropyLoss()
    optimizer = BertAdam(model.parameters(), lr=lr, warmup=warmup, t_total=t_total)
    if state_path is not None:
        init_state = torch.load(state_path)
        model.load_state_dict(init_state['state_dict'])
        optimizer.load_state_dict(init_state['optimizer'])
        init_epoch = init_state['epoch']
    print("Model Config Done.")

    # fine tuning BERT
    print("Start Training.")
    for epoch in range(init_epoch, t_epoch):
        batch_num = 0
        total_loss = 0.0
        for inputs, mask, label in trainloader:
            inputs = inputs.to(device)
            mask = mask.to(device)
            label = label.to(device)
            output = model(inputs, mask)
            loss = criterion(output, label)
            if batch_num%50 == 0:
                print("epoch {}/{}, batch {}/{}, loss = {:.6f}".format(epoch+1, t_epoch, batch_num+1, t_batch, loss.item()))
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch {}/{}, training done. Total loss = {:.6f}".format(epoch+1, t_epoch, total_loss))
        cur_state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(cur_state, "/content/drive/MyDrive/bert/checkpoint/{}_TRAINING_EPOCH_{}.pb".format(task_name, epoch))
        print("epoch {}/{}, model checkpoint saved.".format(epoch+1, t_epoch))
    print("Saving Model......")
    torch.save(model.state_dict(), "/content/drive/MyDrive/bert/checkpoint/{}.pb".format(task_name))
    print("Model saved.")
    print("Training Done.")


if __name__ == "__main__":
    task_name = "IMDB_BERT_Linear_FiT"
    fine_tuning_IMDB(task_name)
