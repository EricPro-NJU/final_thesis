import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
            self.label_idx = torch.LongTensor(label_list)  # num * 2
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
            self.label_idx = torch.LongTensor(label)

    def __len__(self):
        return self.input_idx.shape[0]

    def __getitem__(self, idx):
        return self.input_idx[idx], self.mask_idx[idx], self.label_idx[idx]
