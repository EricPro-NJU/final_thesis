import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer
from server import Log
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

debugging = False

dataset_dict = {
    "IMDB": {
        "train": {
            "source": "/root/autodl-tmp/IMDBtrain.csv",
            "source_type": "csv",
            "source_encoding": "UTF-8-sig",
            "token": "/root/autodl-tmp/IMDBtrain_token.txt",
            "index": "/root/autodl-tmp/IMDBtrain_index.txt",
            "mask": "/root/autodl-tmp/IMDBtrain_mask.txt",
            "label": "/root/autodl-tmp/IMDBtrain_label.txt"
        },
        "test": {
            "source": "/root/autodl-tmp/IMDBtest.csv",
            "source_type": "csv",
            "source_encoding": "UTF-8-sig",
            "token": "/root/autodl-tmp/IMDBtest_token.txt",
            "index": "/root/autodl-tmp/IMDBtest_index.txt",
            "mask": "/root/autodl-tmp/IMDBtest_mask.txt",
            "label": "/root/autodl-tmp/IMDBtest_label.txt"
        },
        "corpus": {
            "source": "/root/autodl-tmp/IMDB_corpus.txt",
            "source_encoding": "UTF-8-sig",
            "token": "/root/autodl-tmp/IMDB_corpus_tokenized.txt",
            "index": "/root/autodl-tmp/IMDB_corpus_indexed.txt"
        },
        "num_class": 2,
        "language": "english",
        "ftp_method": "both"
    },
    "Yelp": {
        "train": {
            "source": "/root/autodl-tmp/Yelptrain.json",
            "source_type": "json",
            "source_encoding": "UTF-8",
            "token": "/root/autodl-tmp/Yelptrain_token.txt",
            "index": "/root/autodl-tmp/Yelptrain_index.txt",
            "mask": "/root/autodl-tmp/Yelptrain_mask.txt",
            "label": "/root/autodl-tmp/Yelptrain_label.txt"
        },
        "test": {
            "source": "/root/autodl-tmp/Yelptest.json",
            "source_type": "json",
            "source_encoding": "UTF-8",
            "token": "/root/autodl-tmp/Yelptest_token.txt",
            "index": "/root/autodl-tmp/Yelptest_index.txt",
            "mask": "/root/autodl-tmp/Yelptest_mask.txt",
            "label": "/root/autodl-tmp/Yelptest_label.txt"
        },
        "corpus": {
            "source": "/root/autodl-tmp/Yelpcorpus.txt",
            "source_encoding": "UTF-8",
            "token": "/root/autodl-tmp/Yelpcorpus_tokenized.txt",
            "index": "/root/autodl-tmp/Yelpcorpus_indexed.txt"
        },
        "num_class": 2,
        "language": "english",
        "ftp_method": "both"
    },
    "AGNews": {
        "train": {
            "source": "/root/autodl-tmp/AGNtrain.json",
            "source_type": "json",
            "source_encoding": "UTF-8",
            "token": "/root/autodl-tmp/AGNtrain_token.txt",
            "index": "/root/autodl-tmp/AGNtrain_index.txt",
            "mask": "/root/autodl-tmp/AGNtrain_mask.txt",
            "label": "/root/autodl-tmp/AGNtrain_label.txt"
        },
        "test": {
            "source": "/root/autodl-tmp/AGNtest.json",
            "source_type": "json",
            "source_encoding": "UTF-8",
            "token": "/root/autodl-tmp/AGNtest_token.txt",
            "index": "/root/autodl-tmp/AGNtest_index.txt",
            "mask": "/root/autodl-tmp/AGNtest_mask.txt",
            "label": "/root/autodl-tmp/AGNtest_label.txt"
        },
        "corpus": {
            "source": "/root/autodl-tmp/AGNcorpus.txt",
            "source_encoding": "UTF-8",
            "token": "/root/autodl-tmp/AGNcorpus_tokenized.txt",
            "index": "/root/autodl-tmp/AGNcorpus_indexed.txt"
        },
        "num_class": 4,
        "language": "english",
        "ftp_method": "masklm"
    },
    "Sogou": {
        "train": {
            "source": "/root/autodl-tmp/Sogoutrain.csv",
            "source_type": "csv",
            "source_encoding": "UTF-8",
            "token": "/root/autodl-tmp/Sogoutrain_token.txt",
            "index": "/root/autodl-tmp/Sogoutrain_index.txt",
            "mask": "/root/autodl-tmp/Sogoutrain_mask.txt",
            "label": "/root/autodl-tmp/Sogoutrain_label.txt"
        },
        "test": {
            "source": "/root/autodl-tmp/Sogoutest.csv",
            "source_type": "csv",
            "source_encoding": "UTF-8",
            "token": "/root/autodl-tmp/Sogoutest_token.txt",
            "index": "/root/autodl-tmp/Sogoutest_index.txt",
            "mask": "/root/autodl-tmp/Sogoutest_mask.txt",
            "label": "/root/autodl-tmp/Sogoutest_label.txt"
        },
        "corpus": {
            "source": "/root/autodl-tmp/Sogoucorpus.txt",
            "source_encoding": "UTF-8",
            "token": "/root/autodl-tmp/Sogoucorpus_tokenized.txt",
            "index": "/root/autodl-tmp/Sogoucorpus_indexed.txt"
        },
        "num_class": 6,
        "language": "chinese",
        "ftp_method": "both"
    },
    "Debugging": {
        "train": {
            "source": "D:/sample_data/source",
            "token": "D:/sample_data/token",
            "index": "D:/sample_data/index",
            "mask": "D:/sample_data/mask",
            "label": "D:/sample_data/label"
        },
        "num_class": 2,
        "language": "english",
        "ftp_method": "both"
    }
}


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction

    ACKNOWLEDGEMENT:
    This is a script from github repo of the package "pytorch_pretrained_bert"
    https://github.com/Meelfy/pytorch_pretrained_BERT/blob/master/examples/run_lm_finetuning.py
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                print("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def index_data(data_path, data_token_path=None, data_index_path=None, data_mask_path=None, data_label_path=None,
               data_type="csv", data_encoding="UTF-8", language='english'):
    '''
    ====================SOURCE DATA FILE FORMAT==========================
    IMDB:
    1. one data for a line
    2. first label (indicated by a single number), then a whitespace or tab, then the whole sentence
    Yelp:
    1. json file
    2. with star, label, text attributes
    =====================================================================
    Read data from file, and save tokens, indexes, mask if necessary
    :param data_path(input_path):
    :param data_token_path(output_path): if None, do not save to file
    :param data_index_path(output_path):
    :param data_mask_path(output_path):
    :param data_label_path(output_path):
    :return: 4 lists including data information
    '''

    data_list = []
    token_list = []
    index_list = []
    mask_list = []
    label_list = []
    length_list = []

    with open(data_path, "r", encoding=data_encoding) as train_file:
        if data_type == 'csv':
            linereader = train_file.readlines()
            for i, line in enumerate(linereader):
                num = eval(line[0])
                sentence = line[2:]
                data_list.append([num, sentence.strip()])
                if debugging and i > 500:
                    break
                if (i + 1) % 10000 == 0:
                    print("Read Data {} / {}".format(i+1, len(linereader)))
        elif data_type == 'json':
            reader = json.load(train_file)
            for i, item in enumerate(reader):
                num = item['label']
                sentence = item['text'].strip()
                data_list.append([num, sentence])
                if debugging and i > 500:
                    break
                if (i + 1) % 10000 == 0:
                    print("Read Data {} / {}".format(i+1, len(reader)))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased' if language == "english" else 'bert-base-chinese')

    for i, item in enumerate(data_list):
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
        length_list.append(tokens_len)
        if (i + 1) % 10000 == 0:
            print("Processed Data {} / {}".format(i + 1, len(data_list)))


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

    return token_list, index_list, mask_list, label_list, length_list


def separate_corpus(corpus_path, save_to=None, data_encoding="UTF-8", language='english'):
    '''
    ====================SOURCE DATA FILE FORMAT===============================================
    1. One sentence for a line
    2. Paragraphs are split by a blank line
    ==========================================================================================
    separate the corpus into two categories: continuous sentences, randoms sentences
    make sure the numbers of items in both lists are the same
    IMPORTANT: items in token list format!
    :param corpus_path(input path):
    :param save_to(output path):
    :return: two lists. list format: token_for_sentence_1, token_for_sentence_2, continuous or not (0 for continuous sentences)
    '''
    print("separating corpus......")
    corpus_list0 = []  # continuous
    corpus_list1 = []  # random
    cache = None
    random_cache = None
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased' if language == "english" else 'bert-base-chinese')
    max_size = 0
    with open(corpus_path, "r", encoding=data_encoding) as fp:
        linereader = fp.readlines()
        file_len = len(linereader)
        for i, line in enumerate(linereader):
            line = line.strip()
            if line != "":
                tokens = tokenizer.tokenize(line)
                if cache is None:
                    cache = tokens
                else:
                    corpus_list0.append([cache, tokens, 0])
                    size = len(cache) + len(tokens)
                    max_size = max_size if max_size > size else size
                    cache = None
            else:
                if cache is not None:
                    if random_cache is not None:
                        corpus_list1.append([cache, random_cache, 1])
                        size = len(cache) + len(random_cache)
                        max_size = max_size if max_size > size else size
                        cache = None
                        random_cache = None
                    else:
                        random_cache = cache
                        cache = None
            if (i + 1) % 2000 == 0:
                print("Read data {} / {}".format(i + 1, file_len))
            if debugging and i > 500:
                break
    len0 = len(corpus_list0)
    len1 = len(corpus_list1)
    while len0 > len1:
        a = random.randint(0, len0 - 1)
        b = random.randint(0, len0 - 1)
        if a == b:
            item = corpus_list0[a]
            temp = item[0]
            item[0] = item[1]
            item[1] = temp
            item[2] = 1
            del corpus_list0[a]
            corpus_list1.append(item)
            len0 -= 1
            len1 += 1
        else:
            itema = corpus_list0[a]
            itemb = corpus_list0[b]
            temp = itema[1]
            itema[1] = itemb[1]
            itemb[1] = temp
            itema[2] = 1
            itemb[2] = 1
            del corpus_list0[a]
            del corpus_list0[b]
            corpus_list1.append(itema)
            corpus_list1.append(itemb)
            len0 -= 2
            len1 += 2
    if save_to is not None:
        with open(save_to, "w", encoding="UTF-8")as fp:
            fp.write("{}\n".format(max_size))
            for item in corpus_list0:
                fp.write("{}\n".format(item))
            for item in corpus_list1:
                fp.write("{}\n".format(item))
    print("Done, max token size is {}, data size {}.".format(max_size, len0+len1))
    return corpus_list0, corpus_list1, max_size


def index_corpus(corpus_path, tokens_path, save_to=None, data_encoding="UTF-8", language='english'):
    '''
    :param corpus_path(input path for planA): same format of input of function separate_corpus
    :param tokens_path(input path for planB): the output format of function separate_corpus
    NOTICE: input plan A has first priority. At least one plan should be identified
    :param save_to(output path):
    :return: # the final indexes should include:
    #   1. input idx with format: [CLS] sentence A [SEP] sentence B [SEP] ([PAD]+) (with words masked)
    #   2. token type idx with format: 0 0 0 ... 0 1 1 1 ... (0 for anything before first [SEP])
    #   3. attention mask idx with format: 1 1 1 ... 1 0 0 0 ... (0 for [PAD])
    #   4. masked lm label with format: second return value of random_words, adding -1 to [PAD],[CLS],[SEP]
    #   5. next sentence label: 0 or 1, 0 for continuous sentences, 1 for random sentences.
    # sequence length is limited to 509+3
    '''

    inputs = []
    token_type = []
    attn_mask = []
    masked_lm = []
    next_sentence = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased' if language == "english" else 'bert-base-chinese')
    if corpus_path is not None:
        list0, list1, max_size = separate_corpus(corpus_path, tokens_path, data_encoding)
        tokens_list = list0 + list1

    else:
        if tokens_path is None:
            raise ValueError("Please assign corpus file path when calling this function")
        tokens_list = []
        max_size = -1
        with open(tokens_path, "r", encoding="UTF-8") as fp:
            lines = fp.readlines()
            file_len = len(lines)
            for i, line in enumerate(lines):
                if max_size == -1:
                    max_size = eval(line.strip())
                    print("read max_size: {}".format(max_size))
                    continue
                tokens_list.append(eval(line.strip()))
                if (i + 1) % 10000 == 0:
                    print("Read data {} / {}".format(i + 1, file_len))
                if debugging and i > 500:
                    break
    if max_size > 128:
        max_size = 128
    # random.shuffle(tokens_list)
    size = len(tokens_list)
    for i, item in enumerate(tokens_list):
        size0 = len(item[0])
        size1 = len(item[1])
        if size0 + size1 > max_size:
            if size0 <= size1:
                if size0 <= max_size // 2:
                    size1 = max_size - size0
                    item[1] = item[1][0:size1]
                else:
                    size0 = max_size // 2
                    size1 = max_size - size0
                    item[0] = item[0][-size0:]
                    item[1] = item[1][0:size1]
            else:
                if size1 <= max_size // 2:
                    size0 = max_size - size1
                    item[0] = item[0][-size0:]
                else:
                    size1 = max_size // 2
                    size0 = max_size - size1
                    item[0] = item[0][-size0:]
                    item[1] = item[1][0:size1]
        total_size = size0 + size1 + 3
        pad_size = max_size + 3 - total_size
        next_sentence.append(item[2])
        tt_item = [0] * (size0 + 1) + [1] * (max_size + 3 - size0 - 1)
        assert len(tt_item) == (max_size + 3)
        token_type.append(tt_item)
        att_item = [1] * total_size + [0] * pad_size
        assert len(att_item) == (max_size + 3)
        attn_mask.append(att_item)
        output_token0, output_label0 = random_word(item[0], tokenizer)
        output_token1, output_label1 = random_word(item[1], tokenizer)
        input_item = ["[CLS]"] + output_token0 + ["[SEP]"] + output_token1 + ["[SEP]"] + ["[PAD]"] * pad_size
        lm_item = [-1] + output_label0 + [-1] + output_label1 + [-1] * (pad_size + 1)
        index_item = tokenizer.convert_tokens_to_ids(input_item)
        assert len(index_item) == (max_size + 3)
        inputs.append(index_item)
        assert len(lm_item) == (max_size + 3)
        masked_lm.append(lm_item)
        if (i + 1) % 1000 == 0:
            print("Processed Data {} / {}".format(i + 1, size))
    if save_to is not None:
        with open(save_to, "w", encoding="UTF-8") as fp:
            for i in range(size):
                temp = [inputs[i], token_type[i], attn_mask[i], masked_lm[i], next_sentence[i]]
                fp.write("{}\n".format(temp))

    return inputs, token_type, attn_mask, masked_lm, next_sentence


def label_logits(labels, group_num):
    logits = []
    for item in labels:
        prob = [0] * group_num
        prob[item] = 1
        logits.append(prob)
    return logits


class TextDataSet(Dataset):
    def __init__(self, name, split="train", read_from_cache=False, log=None, language="english"):
        super(TextDataSet, self).__init__()
        self.name = name
        self.split = split
        self.lg = log
        if name not in dataset_dict:
            self.log("Data not found.")
            raise ValueError("Cannot found dataset named {}".format(name))
        if split not in ["train", "test"]:
            self.log("Split not found.")
            raise ValueError("Cannot found split named {}".format(split))
        src_file = dataset_dict[name][split]["source"]
        token_file = dataset_dict[name][split]["token"]
        index_file = dataset_dict[name][split]["index"]
        mask_file = dataset_dict[name][split]["mask"]
        label_file = dataset_dict[name][split]["label"]
        src_type = dataset_dict[name][split]["source_type"]
        encoding = dataset_dict[name][split]["source_encoding"]
        if read_from_cache:
            self.log("Reading data from cache.")
            index = []
            mask = []
            label = []
            length = []
            with open(index_file, "r", encoding="UTF-8") as fp:
                linereader = fp.readlines()
                for i, line in enumerate(linereader):
                    if debugging and i > 500:
                        break
                    index.append(eval(line))
                    if (i + 1) % 10000 == 0 or (i + 1) == len(linereader):
                        self.log("Reading cache file 1 / 3, Data {} / {}".format(i + 1, len(linereader)))
            with open(mask_file, "r", encoding="UTF-8") as fp:
                linereader = fp.readlines()
                for i, line in enumerate(linereader):
                    if debugging and i > 500:
                        break
                    mask.append(eval(line))
                    length.append(sum(eval(line)))
                    if (i + 1) % 10000 == 0 or (i + 1) == len(linereader):
                        self.log("Reading cache file 2 / 3, Data {} / {}".format(i + 1, len(linereader)))
            with open(label_file, "r", encoding="UTF-8") as fp:
                linereader = fp.readlines()
                for i, line in enumerate(linereader):
                    if debugging and i > 500:
                        break
                    label.append(eval(line))
                    if (i + 1) % 10000 == 0 or (i + 1) == len(linereader):
                        self.log("Reading cache file 3 / 3, Data {} / {}".format(i + 1, len(linereader)))
            self.input_idx = torch.LongTensor(index)
            self.mask_idx = torch.LongTensor(mask)
            self.label_idx = torch.LongTensor(label)
            self.length_idx = torch.LongTensor(length)
        else:
            token_list, index_list, mask_list, label_list, length_list = index_data(src_file, token_file, index_file,
                                                                                    mask_file,
                                                                                    label_file, src_type,
                                                                                    encoding, language)
            self.input_idx = torch.LongTensor(index_list)  # num * seq_len
            self.mask_idx = torch.LongTensor(mask_list)  # num * seq_len
            self.label_idx = torch.LongTensor(label_list)  # num
            self.length_idx = torch.LongTensor(length_list)  # num

    def __len__(self):
        return self.input_idx.shape[0]

    def __getitem__(self, idx):
        return self.input_idx[idx], self.mask_idx[idx], self.label_idx[idx], self.length_idx[idx]

    def log(self, msg):
        if self.lg is None:
            print(msg)
        else:
            self.lg.log(msg)


class TextCorpus(Dataset):
    def __init__(self, name, split="corpus", read_from_cache=False, log=None, language="english"):
        super(TextCorpus, self).__init__()
        self.name = name
        self.split = split
        self.lg = log
        if name not in dataset_dict:
            self.log("Data not found.")
            raise ValueError("Cannot found dataset named {}".format(name))
        if split not in ["corpus"]:
            self.log("Split not found.")
            raise ValueError("Cannot found split named {}".format(split))
        src_file = dataset_dict[name][split]["source"]
        token_file = dataset_dict[name][split]["token"]
        index_file = dataset_dict[name][split]["index"]
        encoding = dataset_dict[name][split]["source_encoding"]
        if read_from_cache:
            self.log("Reading data from cache file.")
            inputs = []
            tokentype = []
            attn = []
            masklm = []
            nextsen = []
            with open(index_file, "r", encoding="UTF-8") as fp:
                lines = fp.readlines()
                file_size = len(lines)
                for i, line in enumerate(lines):
                    temp = eval(line.strip())
                    inputs.append(temp[0])
                    tokentype.append(temp[1])
                    attn.append(temp[2])
                    masklm.append(temp[3])
                    nextsen.append(temp[4])
                    if (i + 1) % 10000 == 0 or (i + 1) == file_size:
                        self.log("Reading cache file 1 / 1, data {} / {}".format(i + 1, file_size))
                    if debugging and (i + 1) >= 500:
                        break
        else:
            inputs, tokentype, attn, masklm, nextsen = index_corpus(src_file, token_file, index_file,
                                                                    encoding, language)
        self.input_idx = torch.LongTensor(inputs)
        self.token_type = torch.LongTensor(tokentype)
        self.attn_mask = torch.LongTensor(attn)
        self.masked_lm = torch.LongTensor(masklm)
        self.next_sentence = torch.LongTensor(nextsen)

    def __len__(self):
        return self.input_idx.shape[0]

    def __getitem__(self, idx):
        return self.input_idx[idx], self.token_type[idx], self.attn_mask[idx], self.masked_lm[idx], self.next_sentence[
            idx]

    def log(self, msg):
        if self.lg is None:
            print(msg)
        else:
            self.lg.log(msg)


if __name__ == "__main__":
    '''
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    sentence = "Behold! the abyss had come. Let the flame of redemption rage!"
    tokens = tokenizer.tokenize(sentence)
    print(tokens)
    output_tokens, output_label = random_word(tokens, tokenizer)
    print(output_tokens)
    print(output_label)
    '''
    trainset = TextDataSet('AGNews', split='train')
    testset = TextDataSet('AGNews', split='test')
    corpusset = TextCorpus('AGNews')
