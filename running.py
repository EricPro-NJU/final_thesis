import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import time
import server
from pytorch_pretrained_bert import BertAdam, BertForPreTraining
from datasets import IMDBDataSet, IMDBCorpus
from bert import SimpleBert, RecBert
from transformer import TransformerEncoder
from basis import SimpleLSTM
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

push_message = True


class Log:
    def __init__(self, task_name):
        self.task_name = task_name
        self.log_list = []
        self.log_path = "/root/autodl-nas/log/{}_{}.log".format(task_name,
                                                                time.strftime("%Y%m%d%H%M%S", time.localtime()))
        self.log_num = 0

    def log(self, text, mute=False, discard=False, message=False):
        log_text = "{}\t{}".format(time.strftime("LOG: %Y-%m-%d %H:%M:%S", time.localtime()), text)
        if not discard:
            self.log_list.append("{}\n".format(log_text))
            self.log_num += 1
        if not mute:
            print(log_text)
        if message:
            server.send_log_message(self.task_name, text)

    def writelog(self):
        if self.log_num == 0:
            print("Warning from class Log: No log will be written.")
            return
        with open(self.log_path, "w", encoding="UTF-8") as fp:
            fp.writelines(self.log_list)
        self.log_list.clear()
        self.log_num = 0
        self.log_path = "/root/autodl-nas/log/{}_{}.log".format(task_name,
                                                                time.strftime("%Y%m%d%H%M%S", time.localtime()))


def further_pretraining(task_name, datasets="IMDB", batch_size=32, state_path=None):
    # datasets should be read using Dataset class into Dataloader
    # use uncased BERT pretraining model to further pretrain the model
    # set checkpoint each epoch trained
    # finally save the "BertModel"(model.bert) state_dict to local files, this could be read in fine_tuning
    torch.cuda.empty_cache()
    lg = Log(task_name)
    # load training data and indexing texts
    lg.log("Indexing Training Data......")

    if datasets == "IMDB":
        corpus_file = "/root/autodl-tmp/IMDB_corpus.txt"
        corpus_token_file = "/root/autodl-tmp/IMDB_corpus_tokenized.txt"
        corpus_index_file = "/root/autodl-tmp/IMDB_corpus_indexed.txt"
        dataloader = DataLoader(IMDBCorpus(None, None, corpus_index_file), batch_size=batch_size,
                                shuffle=True)
    else:
        raise ValueError("No such dataset called {}".format(datasets))
    t_batch = len(dataloader)
    lg.log("Index Training Data Done.")

    # prepare further_pretraining model
    lg.log("Model Config......")
    model = BertForPreTraining.from_pretrained("bert-base-uncased").to(device)
    lg.log("BertForPreTraining loaded.")
    init_epoch = 0
    t_epoch = 10
    lr = 5e-5
    warmup = 0.1
    t_total = 1e5
    optimizer = BertAdam(model.parameters(), lr=lr, warmup=warmup, t_total=t_total)
    if state_path is not None:
        init_state = torch.load(state_path)
        model.load_state_dict(init_state['state_dict'])
        optimizer.load_state_dict(init_state['optimizer'])
        init_epoch = init_state['epoch'] + 1
        lg.log("Read model checkpoint in epoch {}. Training will be initiated from epoch {}".format(init_epoch,
                                                                                                    init_epoch + 1))
    lg.log("Model Config Done.")

    # pretraining BERT
    lg.log("Start PreTraining.")
    start_time = time.time()
    last_time = start_time
    for epoch in range(init_epoch, t_epoch):
        batch_num = 0
        total_loss = 0.0
        for inputs, ttype, mask, lm, nxtsen in dataloader:
            inputs = inputs.to(device)
            ttype = ttype.to(device)
            mask = mask.to(device)
            lm = lm.to(device)
            nxtsen = nxtsen.to(device)
            loss = model(inputs, ttype, mask, lm, nxtsen)
            if (batch_num + 1) % 50 == 0:
                lg.log("epoch {}/{}, batch {}/{}, loss = {:.6f}".format(epoch + 1, t_epoch, batch_num + 1, t_batch,
                                                                        loss.item()))
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_num += 1

        this_time = time.time()
        lg.log(
            "epoch {}/{}, training done. Average loss = {:.6f}, Time Elapse this epoch : {}".format(epoch + 1, t_epoch,
                                                                                                    total_loss / t_batch,
                                                                                                    time.strftime(
                                                                                                        "%H:%M:%S",
                                                                                                        time.gmtime(
                                                                                                            this_time - last_time))),
            message=push_message)
        cur_state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(cur_state, "/root/autodl-tmp/checkpoint/{}_TRAINING_EPOCH_{}.pb".format(task_name, epoch))
        lg.log("epoch {}/{}, model checkpoint saved.".format(epoch + 1, t_epoch))
        last_time = time.time()

    lg.log("Saving Pretraining Model......")
    torch.save(model.bert.state_dict(), "/root/autodl-nas/checkpoint/{}.pb".format(task_name))
    lg.log("Pretraining Model saved.")
    final_time = time.time()
    lg.log("Training Done. Time elapsed: {}".format(time.strftime("%H:%M:%S", time.gmtime(final_time - start_time))),
           message=push_message)
    lg.writelog()


def basis_training(task_name, datasets="IMDB", batch_size=24, model_name="sp_lstm", bidirec=True, state_path=None):
    torch.cuda.empty_cache()
    lg = Log(task_name)
    # load training data and indexing texts
    lg.log("Indexing Training Data......")

    if datasets == "IMDB":
        train_file = "/root/autodl-tmp/IMDBtrain.csv"
        train_token_file = "/root/autodl-tmp/IMDBtrain_token.txt"
        train_index_file = "/root/autodl-tmp/IMDBtrain_index.txt"
        train_mask_file = "/root/autodl-tmp/IMDBtrain_mask.txt"
        train_label_file = "/root/autodl-tmp/IMDBtrain_label.txt"
        trainloader = DataLoader(
            IMDBDataSet(None, train_token_file, train_index_file, train_mask_file, train_label_file),
            batch_size=batch_size, shuffle=True)
    else:
        raise ValueError("No such dataset called {}".format(datasets))
    t_batch = len(trainloader)
    lg.log("Index Training Data Done.")

    # prepare model and set hyper params
    lg.log("Model Config......")
    if model_name == "sp_lstm":
        model = SimpleLSTM(512, 1024, 2, bidirec=bidirec).to(device)
        lg.log("choosing Simple {}LSTM model.".format("bi-directional " if bidirec else ""))
    else:
        raise ValueError("No such model named {}.".format(model_name))
    model.train()

    init_epoch = 0
    t_epoch = 4
    lr = 1e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if state_path is not None:
        init_state = torch.load(state_path)
        model.load_state_dict(init_state['state_dict'])
        optimizer.load_state_dict(init_state['optimizer'])
        init_epoch = init_state['epoch'] + 1
        lg.log("Read model checkpoint in epoch {}. Training will be initiated from epoch {}".format(init_epoch,
                                                                                                    init_epoch + 1))
    lg.log("Model Config Done.")

    # training
    lg.log("Start Training.")
    start_time = time.time()
    last_time = start_time
    for epoch in range(init_epoch, t_epoch):
        batch_num = 0
        total_loss = 0.0
        for inputs, mask, label in trainloader:
            inputs = inputs.to(device)
            label = label.to(device)
            output = model(inputs)
            # N * output_size (after softmax, represent probability)  eg. N * 2
            loss = criterion(output, label)
            if (batch_num + 1) % 50 == 0:
                lg.log("epoch {}/{}, batch {}/{}, loss = {:.6f}".format(epoch + 1, t_epoch, batch_num + 1, t_batch,
                                                                        loss.item()))
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_num += 1

        this_time = time.time()
        lg.log(
            "epoch {}/{}, training done. Average loss = {:.6f}, Time Elapse this epoch : {}".format(epoch + 1, t_epoch,
                                                                                                    total_loss / t_batch,
                                                                                                    time.strftime(
                                                                                                        "%H:%M:%S",
                                                                                                        time.gmtime(
                                                                                                            this_time - last_time))),
            message=push_message)
        cur_state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(cur_state, "/root/autodl-tmp/checkpoint/{}_TRAINING_EPOCH_{}.pb".format(task_name, epoch))
        lg.log("epoch {}/{}, model checkpoint saved.".format(epoch + 1, t_epoch))
        last_time = time.time()

    lg.log("Saving Model......")
    torch.save(model.state_dict(), "/root/autodl-nas/checkpoint/{}.pb".format(task_name))
    lg.log("Model saved.")
    final_time = time.time()
    lg.log("Training Done. Time elapsed: {}".format(time.strftime("%H:%M:%S", time.gmtime(final_time - start_time))),
           message=push_message)
    lg.writelog()


def fine_tuning(task_name, datasets="IMDB", batch_size=16, model_name="linear", bidirec=True,
                further_pretrained=None, state_path=None):
    torch.cuda.empty_cache()
    lg = Log(task_name)
    # load training data and indexing texts
    lg.log("Indexing Training Data......")

    if datasets == "IMDB":
        train_file = "/root/autodl-tmp/IMDBtrain.csv"
        train_token_file = "/root/autodl-tmp/IMDBtrain_token.txt"
        train_index_file = "/root/autodl-tmp/IMDBtrain_index.txt"
        train_mask_file = "/root/autodl-tmp/IMDBtrain_mask.txt"
        train_label_file = "/root/autodl-tmp/IMDBtrain_label.txt"
        trainloader = DataLoader(
            IMDBDataSet(None, train_token_file, train_index_file, train_mask_file, train_label_file),
            batch_size=batch_size, shuffle=True)
    else:
        raise ValueError("No such dataset called {}".format(datasets))
    t_batch = len(trainloader)
    lg.log("Index Training Data Done.")

    # prepare BERT model and set hyper params
    lg.log("Model Config......")
    if model_name == "linear":
        model = SimpleBert(512, 2).to(device)
        lg.log("choosing BERT + Linear model.")
    elif model_name == "lstm":
        model = RecBert(512, 1024, 2, bidirec).to(device)
        lg.log("choosing BERT + {}LSTM model.".format("bi-directional " if bidirec else ""))
    else:
        model = SimpleBert(512, 2).to(device)
        lg.log("WARNING!! No implemented model called {}. Use default setting instead.".format(model_name))
        lg.log("choosing BERT + Linear model.")
    model.train()

    init_epoch = 0
    t_epoch = 4
    lr = 2e-5

    criterion = nn.CrossEntropyLoss()
    optimizer = BertAdam(model.parameters(), lr=lr)
    if state_path is not None:
        init_state = torch.load(state_path)
        model.load_state_dict(init_state['state_dict'])
        optimizer.load_state_dict(init_state['optimizer'])
        init_epoch = init_state['epoch'] + 1
        lg.log("Read model checkpoint in epoch {}. Training will be initiated from epoch {}".format(init_epoch,
                                                                                                    init_epoch + 1))
    elif further_pretrained:
        model.bert.load_state_dict(torch.load(further_pretrained))
        lg.log("Read further_pretrained model from file.")
    lg.log("Model Config Done.")

    # fine tuning BERT
    lg.log("Start Training.")
    start_time = time.time()
    last_time = start_time
    for epoch in range(init_epoch, t_epoch):
        batch_num = 0
        total_loss = 0.0
        for inputs, mask, label in trainloader:
            inputs = inputs.to(device)
            mask = mask.to(device)
            label = label.to(device)
            output = model(inputs, mask)
            # N * output_size (after softmax, represent probability)  eg. N * 2
            loss = criterion(output, label)
            if (batch_num + 1) % 50 == 0:
                lg.log("epoch {}/{}, batch {}/{}, loss = {:.6f}".format(epoch + 1, t_epoch, batch_num + 1, t_batch,
                                                                        loss.item()))
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_num += 1

        this_time = time.time()
        lg.log(
            "epoch {}/{}, training done. Average loss = {:.6f}, Time Elapse this epoch : {}".format(epoch + 1, t_epoch,
                                                                                                    total_loss / t_batch,
                                                                                                    time.strftime(
                                                                                                        "%H:%M:%S",
                                                                                                        time.gmtime(
                                                                                                            this_time - last_time))),
            message=push_message)
        cur_state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(cur_state, "/root/autodl-tmp/checkpoint/{}_TRAINING_EPOCH_{}.pb".format(task_name, epoch))
        lg.log("epoch {}/{}, model checkpoint saved.".format(epoch + 1, t_epoch))
        last_time = time.time()

    lg.log("Saving Model......")
    torch.save(model.state_dict(), "/root/autodl-nas/checkpoint/{}.pb".format(task_name))
    lg.log("Model saved.")
    final_time = time.time()
    lg.log("Training Done. Time elapsed: {}".format(time.strftime("%H:%M:%S", time.gmtime(final_time - start_time))),
           message=push_message)
    lg.writelog()


def evaluate(task_name, model_path, datasets="IMDB", batch_size=24, model_name="linear", bidirec=True):
    torch.cuda.empty_cache()
    lg = Log(task_name)
    # load testing data and indexing texts
    lg.log("Indexing Testing Data......")

    if datasets == "IMDB":
        test_file = "/root/autodl-tmp/IMDBtest.csv"
        test_token_file = "/root/autodl-tmp/IMDBtest_token.txt"
        test_index_file = "/root/autodl-tmp/IMDBtest_index.txt"
        test_mask_file = "/root/autodl-tmp/IMDBtest_mask.txt"
        test_label_file = "/root/autodl-tmp/IMDBtest_label.txt"
        testloader = DataLoader(IMDBDataSet(None, test_token_file, test_index_file, test_mask_file, test_label_file),
                                batch_size=batch_size, shuffle=True)
    else:
        raise ValueError("No such dataset called {}".format(datasets))
    t_batch = len(testloader)
    lg.log("Index Testing Data Done.")

    # prepare BERT model and set hyper params
    lg.log("Model Config......")
    if model_name == "linear":
        model = SimpleBert(512, 2).to(device)
        lg.log("choosing BERT + Linear model.")
    elif model_name == "lstm":
        model = RecBert(512, 1024, 2, bidirec).to(device)
        lg.log("choosing BERT + {}LSTM model.".format("bi-directional " if bidirec else ""))
    elif model_name == "sp_lstm":
        model = SimpleLSTM(512, 1024, 2, bidirec=bidirec).to(device)
        lg.log("choosing Simple {}LSTM model.".format("bi-directional " if bidirec else ""))

    else:
        model = SimpleBert(512, 2).to(device)
        lg.log("WARNING!! No implemented model called {}. Use default setting instead.".format(model_name))
        lg.log("choosing BERT + Linear model.")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # evaluate
    lg.log("Testing......")
    val_loss = 0.0
    val_total = 0
    val_cor = 0

    batch_num = 0
    with torch.no_grad():
        for inputs, mask, label in testloader:
            inputs = inputs.to(device)
            mask = mask.to(device)
            label = label.to(device)
            output = model(inputs, mask)
            # N * output_size (after softmax, represent probability)  eg. N * 2
            loss = criterion(output, label)
            val_loss += loss.item()

            prediction = output.argmax(dim=-1)
            answer = label.view(-1)
            val_total += prediction.shape[0]
            val_cor += prediction[prediction == answer].shape[0]
            if (batch_num + 1) % 50 == 0:
                lg.log("Testing {} / {} done.".format(batch_num + 1, t_batch))
            batch_num += 1

    val_loss = val_loss / t_batch
    acc = val_cor / val_total
    lg.log("Test Result: {} / {} correct, {} accuracy, {} average loss.".format(val_cor, val_total, acc, val_loss),
           message=push_message)
    lg.writelog()


def ft1():
    print("INITIATING TASK: IMDB_BERTLN_FtP10k_FiT")
    task_name = "IMDB_BERTLN_FtP100k_FiT"
    further_pretrained = "/root/autodl-nas/checkpoint/IMDB_FtP.pb"
    fine_tuning(task_name, datasets="IMDB", model_name="linear", further_pretrained=further_pretrained,
                batch_size=16)
    model_path = "/root/autodl-nas/checkpoint/IMDB_BERTLN_FtP100k_FiT.pb"
    evaluate(task_name, model_path=model_path, datasets="IMDB", model_name="linear")


def ft2():
    print("INITIATING TASK: IMDB_BERTRNN_FtP10k_FiT")
    task_name = "IMDB_BERTRNN_FtP100k_FiT"
    further_pretrained = "/root/autodl-nas/checkpoint/IMDB_FtP.pb"
    fine_tuning(task_name, datasets="IMDB", model_name="lstm", further_pretrained=further_pretrained, batch_size=16)
    model_path = "/root/autodl-nas/checkpoint/IMDB_BERTRNN_FtP100k_FiT.pb"
    evaluate(task_name, model_path=model_path, datasets="IMDB", model_name="lstm")


def fp():
    print("INITIATING TASK: IMDB_FtP")
    task_name = "IMDB_FtP"
    further_pretraining(task_name, datasets="IMDB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Initiate the task by inputting like: 'python3 running.py [task_name]', eg. 'python3 running.py ft1'.")
        raise ValueError("No task name assigned")
    if sys.argv[1] == "ft1":
        ft1()
    elif sys.argv[1] == "ft2":
        ft2()
    elif sys.argv[1] == "fp":
        fp()
    elif sys.argv[1] == "ft":
        ft1()
        ft2()
    else:
        raise ValueError("Can't read task name.")
