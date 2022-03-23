import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import time
from server import Log
from pytorch_pretrained_bert import BertAdam, BertForPreTraining
from datasets import TextDataSet, TextCorpus, dataset_dict
import datasets
from bert import SimpleBert, RecBert
from transformer import TransformerEncoder
from basis import TextRNN, TextCNN
import sys
import argparse
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

push_message = False

model_dict = {"textrnn", "textcnn", "transformer", "bert_linear", "bert_lstm"}


def further_pretraining(task_name, datasets="IMDB", batch_size=32, state_path=None, read_from_cache="False"):
    # datasets should be read using Dataset class into Dataloader
    # use uncased BERT pretraining model to further pretrain the model
    # set checkpoint each epoch trained
    # finally save the "BertModel"(model.bert) state_dict to local files, this could be read in fine_tuning
    torch.cuda.empty_cache()
    lg = Log(task_name)
    # load training data and indexing texts
    lg.log("Indexing Training Data......")
    if datasets in dataset_dict:
        dataloader = DataLoader(
            TextCorpus(datasets, split="corpus", read_from_cache=read_from_cache, log=lg),
            batch_size=batch_size,
            shuffle=True
        )
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


def basis_training(task_name, datasets="IMDB", batch_size=24, model_name="sp_lstm", state_path=None,
                   read_from_cache=False):
    torch.cuda.empty_cache()
    lg = Log(task_name)
    # load training data and indexing texts
    lg.log("Indexing Training Data......")
    if datasets in dataset_dict:
        trainloader = DataLoader(
            TextDataSet(datasets, split="train", read_from_cache=read_from_cache, log=lg),
            batch_size=batch_size,
            shuffle=True
        )
    else:
        raise ValueError("No such dataset called {}".format(datasets))
    t_batch = len(trainloader)
    num_class = dataset_dict[datasets]["num_class"]
    lg.log("Index Training Data Done.")

    # prepare model and set hyper params
    lg.log("Model Config......")
    if model_name == "textrnn":
        model = TextRNN(512, 1024, num_class).to(device)
        lg.log("choosing {}TextRNN model.".format("bi-directional "))
    elif model_name == "textcnn":
        model = TextCNN(512, 8, num_class, (5, 5)).to(device)
        lg.log("choosing TextCNN model.")
    else:
        raise ValueError("No such model named {}.".format(model_name))
    model.train()

    init_epoch = 0
    t_epoch = 10
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
        for inputs, mask, label, length in trainloader:
            inputs = inputs.to(device)
            label = label.to(device)
            length = length.to(device)
            output = model(inputs) if model_name == "textcnn" else output = model(inputs, length)
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


def fine_tuning(task_name, datasets="IMDB", batch_size=16, model_name="linear",
                further_pretrained=None, state_path=None, read_from_cache=False):
    torch.cuda.empty_cache()
    lg = Log(task_name)
    # load training data and indexing texts
    lg.log("Indexing Training Data......")

    if datasets in dataset_dict:
        trainloader = DataLoader(
            TextDataSet(datasets, split="train", read_from_cache=read_from_cache, log=lg),
            batch_size=batch_size, shuffle=True)
    else:
        raise ValueError("No such dataset called {}".format(datasets))
    t_batch = len(trainloader)
    num_class = dataset_dict[datasets]["num_class"]
    lg.log("Index Training Data Done.")

    # prepare BERT model and set hyper params
    lg.log("Model Config......")
    if model_name == "bert_linear":
        model = SimpleBert(512, num_class).to(device)
        lg.log("choosing BERT + Linear model.")
    elif model_name == "bert_lstm":
        model = RecBert(512, 1024, num_class).to(device)
        lg.log("choosing BERT + {}LSTM model.".format("bi-directional "))
    else:
        model = SimpleBert(512, num_class).to(device)
        lg.log("WARNING!! No implemented model called {}. Use default setting instead.".format(model_name))
        lg.log("choosing BERT + Linear model.")
    model.train()

    init_epoch = 0
    t_epoch = 4
    lr = 2e-5
    t_total = t_batch * t_epoch
    warmup = 0.1

    criterion = nn.CrossEntropyLoss()
    optimizer = BertAdam(model.parameters(), lr=lr, t_total=t_total, warmup=warmup)
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
        for inputs, mask, label, _ in trainloader:
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


def evaluate(task_name, model_path, datasets="IMDB", batch_size=24, model_name="linear", read_from_cache=False):
    torch.cuda.empty_cache()
    lg = Log(task_name)
    # load testing data and indexing texts
    lg.log("Indexing Testing Data......")

    if datasets in dataset_dict:
        testloader = DataLoader(
            TextDataSet(datasets, split="test", read_from_cache=read_from_cache, log=lg),
            batch_size=batch_size, shuffle=True)
    else:
        raise ValueError("No such dataset called {}".format(datasets))
    t_batch = len(testloader)
    num_class = dataset_dict[datasets]["num_class"]
    lg.log("Index Testing Data Done.")

    # prepare BERT model and set hyper params
    lg.log("Model Config......")
    if model_name == "bert_linear":
        model = SimpleBert(512, num_class).to(device)
        lg.log("choosing BERT + Linear model.")
    elif model_name == "bert_lstm":
        model = RecBert(512, 1024, num_class).to(device)
        lg.log("choosing BERT + {}LSTM model.".format("bi-directional "))
    elif model_name == "textrnn":
        model = TextRNN(512, 1024, num_class).to(device)
        lg.log("choosing {}TextRNN model.".format("bi-directional "))
    elif model_name == "textcnn":
        model = TextCNN(512, 8, num_class, (5, 5)).to(device)
        lg.log("choosing TextCNN model.")
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
        for inputs, mask, label, length in testloader:
            inputs = inputs.to(device)
            mask = mask.to(device)
            label = label.to(device)
            length = length.to(device)
            output = model(inputs, mask) if model_name in ["bert_linear", "bert_lstm"] else \
                (model(inputs) if model_name == "textcnn" else model(inputs, length))
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


# ======================TRAINING SCRIPTS=========================


def valid(args):
    if args.data not in dataset_dict:
        return 2, "Dataset not found ({} is not in the dataset dict)".format(args.data)
    if args.model not in model_dict:
        return 2, "Model not found ({} is not in the model dict)".format(args.model)
    if args.further_pretraining and args.model not in ["bert_linear", "bert_lstm"]:
        return 2, "Further pretraining can only perform in Bert Models."
    if args.fine_tuning and args.model not in ["bert_linear", "bert_lstm"]:
        return 2, "Fine tuning can only perform in Bert Models."
    if args.training and args.model in ["bert_linear", "bert_lstm"]:
        return 2, "Please use fine tuning instead of training for Bert Models."
    if args.testing:
        if not (args.fine_tuning or args.training):
            if not args.test_model_path:
                return 2, "No training configs or identify any testing model path."
    if args.fine_tuning and args.further_pretraining:
        if args.fit_ftp_path:
            return 1, "WARNING: You config to further pretrain Bert model but still identify a FtP Model path. The " \
                      "new Ftp Model will be saved, but not utilized for fine-tuning! "
    if args.testing and (args.fine_tuning or args.training):
        if args.test_model_path:
            return 1, "WARNING: You config to train but still identify a saved model path. The newly trained model " \
                      "will be saved, but never utilized for testing!"
    if args.further_pretraining:
        save_to = "/root/autodl-nas/checkpoint/{}_FtP.pb".format(args.name)
        if os.path.exists(save_to):
            return 1, "WARNING: Further pretraining model {} exists. The newly trained model will overwritten!".format(
                save_to)
    if args.fine_tuning or args.training:
        save_to = "/root/autodl-nas/checkpoint/{}.pb".format(args.name)
        if os.path.exists(save_to):
            return 1, "WARNING: Training model {} exists. The newly trained model will overwritten!".format(save_to)
    if not args.read_from_cache:
        return 1, "WARNING: Data read from source file instead of cache. It may rewritten data in cache files! We " \
                  "urge you to use read_from_cache modes especially in debugging mode! "
    return 0, "Parameter validation check pass"


def info(args):
    print("============ERIC'S FINAL THESIS MODEL TRAINING SYSTEM==============")
    print("Config Info: ")
    phase = 1
    if args.debug:
        print("Debugging Mode activated.")
    if args.further_pretraining:
        print("Phase {}: Further Pretraining Bert Model with Standard Pretraining Tasks.".format(phase))
        print("    Task name: {}_FtP".format(args.name))
        print("    Training dataset: {}".format(args.data))
        print("    Batch Size: {}".format(args.ftp_batch_size))
        print("    State Path: {}".format(args.ftp_state_path if args.ftp_state_path else "Not Indicated"))
        print("    Data Read from Cache: {}".format("Yes" if args.read_from_cache else "No"))
        print("    Model Save to: {}".format("/root/autodl-nas/checkpoint/{}_FtP.pb".format(args.name)))
        phase += 1
    if args.fine_tuning:
        print("Phase {}: Fine Tuning Bert Model.".format(phase))
        print("    Task name: {}".format(args.name))
        print("    Training dataset: {}".format(args.data))
        print("    Model: {}".format(args.model))
        print("    Batch Size: {}".format(args.fit_batch_size))
        print("    State Path: {}".format(args.fit_state_path if args.fit_state_path else "Not Indicated"))
        print(
            "    Pretrained Model: {}".format(args.fit_ftp_path if args.fit_ftp_path else "Uncased or from last phase"))
        print("    Data Read from Cache: {}".format("Yes" if args.read_from_cache else "No"))
        print("    Model Save to: {}".format("/root/autodl-nas/checkpoint/{}.pb".format(args.name)))
        phase += 1
    if args.training:
        print("Phase {}: Training Model.".format(phase))
        print("    Task name: {}".format(args.name))
        print("    Training dataset: {}".format(args.data))
        print("    Model: {}".format(args.model))
        print("    Batch Size: {}".format(args.train_batch_size))
        print("    State Path: {}".format(args.train_state_path if args.train_state_path else "Not Indicated"))
        print("    Data Read from Cache: {}".format("Yes" if args.read_from_cache else "No"))
        print("    Model Save to: {}".format("/root/autodl-nas/checkpoint/{}.pb".format(args.name)))
        phase += 1
    if args.testing:
        print("Phase {}: Testing Model.".format(phase))
        print("    Task name: {}".format(args.name))
        print("    Testing dataset: {}".format(args.data))
        print("    Model: {}".format(args.model))
        print("    Batch Size: {}".format(args.test_batch_size))
        print("    Trained Model: {}".format(args.test_model_path if args.test_model_path else "From last phase"))
        print("    Data Read from Cache: {}".format("Yes" if args.read_from_cache else "No"))


def session(args):
    if args.alarm:
        push_message = True
    if args.debug:
        datasets.debugging = True
    if args.further_pretraining:
        task_name = "{}_FtP".format(args.name)
        further_pretraining(task_name, args.data, args.ftp_batch_size, args.ftp_state_path, args.read_from_cache)
    if args.fine_tuning:
        if args.further_pretraining:
            if args.fit_ftp_path:
                ftp_path = args.fit_ftp_path
            else:
                ftp_path = "/root/autodl-nas/checkpoint/{}_FtP.pb".format(args.name)
        else:
            ftp_path = args.fit_ftp_path
        fine_tuning(args.name, args.data, args.fit_batch_size, args.model, ftp_path, args.fit_state_path,
                    args.read_from_cache)
    if args.training:
        basis_training(args.name, args.data, args.train_batch_size, args.model, args.train_state_path,
                       args.read_from_cache)
    if args.testing:
        if args.fine_tuning or args.training:
            if args.test_model_path:
                model_path = args.test_model_path
            else:
                model_path = "/root/autodl-nas/checkpoint/{}.pb".format(args.name)
        else:
            model_path = args.test_model_path
        evaluate(args.name, model_path, args.data, args.test_batch_size, args.model, args.read_from_cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument("--name", "-n", help="Identify task name", required=True, type=str)
    parser.add_argument("--data", "-d", help="Indicate the name of dataset", required=True, type=str)
    parser.add_argument("--model", "-m", help="Indicate the name of model", required=True, type=str)
    # task phase indicate
    parser.add_argument("--further_pretraining", "-ftp", help="Do further pretraining on given corpus (For Bert "
                                                              "related model only)", action="store_true")
    parser.add_argument("--fine_tuning", "-fit", help="Do Fine Tuning on given training data (For Bert related model "
                                                      "only", action="store_true")
    parser.add_argument("--training", "-tr", help="Do Training (For baseline model only", action="store_true")
    parser.add_argument("--testing", "-ev", help="Do Testing on given model", action="store_true")
    # custom settings
    parser.add_argument("--read_from_cache", help="Read data from cache file processed early",
                        action="store_true")
    parser.add_argument("--ftp_batch_size", help="Batch size for further pretraining", type=int, default=32)
    parser.add_argument("--fit_batch_size", help="Batch size for fine tuning", type=int, default=16)
    parser.add_argument("--train_batch_size", help="Batch size for training", type=int, default=32)
    parser.add_argument("--test_batch_size", help="Batch size for testing", type=int, default=32)
    # load state_path
    parser.add_argument("--ftp_state_path", help="Load state of further_pretraining", type=str, default=None)
    parser.add_argument("--fit_state_path", help="Load state of fine_tuning", type=str, default=None)
    parser.add_argument("--train_state_path", help="Load state of training", type=str, default=None)
    parser.add_argument("--fit_ftp_path", help="Load args of further_pretrained Bert model", type=str, default=None)
    parser.add_argument("--test_model_path", help="Load model for testing, default'/root/autodl-nas/checkpoint/["
                                                  "--name].pb'", default=None)
    parser.add_argument("--debug", help="Debug Mode (Only read a small number of data from dataset",
                        action="store_true")
    parser.add_argument("--alarm", help="Alarm when epochs of training done, or testing done. Send messages to Wechat",
                        action="store_true")
    print("Parsing arguments......")
    args = parser.parse_args()
    ret, msg = valid(args)
    if ret == 2:
        raise ValueError(msg)
    elif ret == 1:
        print("\033[33m{}\033[0m".format(msg))
        ans = input("Your input has received warning, do you still want to initiate your session?(Y/N)")
        if ans != "Y" and ans != "y":
            raise ValueError("Session terminated.")
    else:
        print(msg)
    info(args)
    print("Start Session.")
    session(args)
