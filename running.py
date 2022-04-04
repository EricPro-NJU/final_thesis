import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import time
from server import Log
from pytorch_pretrained_bert import BertAdam, BertForPreTraining, BertForMaskedLM
from datasets import TextDataSet, TextCorpus, dataset_dict
import datasets
from bert import SimpleBert, RecBert
from basis import TextRNN, TextCNN, TransformerClassifier
import sys
import argparse
import os
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

push_message = False

model_dict = {"textrnn", "textcnn", "transformer", "bert_linear", "bert_lstm", "bert_lstm2", "bert_lstm3"}
bert_dict = {"bert_linear", "bert_lstm", "bert_lstm2", "bert_lstm3"}

def f1_count(tf_matrix, label_count, prediction_count, lg):
    num_class = label_count.shape[0]
    lg.log("TF MATRIX TABLE\n{}".format(tf_matrix))
    tp = torch.zeros([num_class])
    fp = torch.zeros([num_class])
    fn = torch.zeros([num_class])
    for i in range(num_class):
        tp[i] = tf_matrix[i][i]
        fp[i] = prediction_count[i] - tp[i]
        fn[i] = label_count[i] - tp[i]
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    macro_p = p.mean()
    macro_r = r.mean()
    macro_f1 = (2 * macro_p * macro_r) / (macro_p + macro_r)
    micro_p = tp.mean() / (tp.mean() + fp.mean())
    micro_r = tp.mean() / (tp.mean() + fn.mean())
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r)
    return macro_f1.item(), micro_f1.item()


def further_pretraining(task_name, datasets="IMDB", batch_size=32, state_path=None, read_from_cache="False",
                        language="english", method="both"):
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
            TextCorpus(datasets, split="corpus", read_from_cache=read_from_cache, log=lg, language=language),
            batch_size=batch_size,
            shuffle=True
        )
    else:
        raise ValueError("No such dataset called {}".format(datasets))

    t_batch = len(dataloader)
    lg.log("Index Training Data Done.")

    # prepare further_pretraining model
    load = 'bert-base-uncased' if language == "english" else 'bert-base-chinese'
    lg.log("Model Config......")
    if method == "both":
        model = BertForPreTraining.from_pretrained(load).to(device)
    elif method == "masklm":
        model = BertForMaskedLM.from_pretrained(load).to(device)
    else:
        raise ValueError("Invalid training method.")
    lg.log("BertForPreTraining loaded.")
    init_epoch = 0
    t_epoch = math.ceil(1e5 / t_batch)
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
    step = 0 + t_batch * init_epoch
    # pretraining BERT
    lg.log("Start PreTraining.")
    start_time = time.time()
    last_time = start_time
    for epoch in range(init_epoch, t_epoch):
        if step > t_total:
            lg.log("Reached step limit, epoch skipped.")
            break
        batch_num = 0
        total_loss = 0.0
        for inputs, ttype, mask, lm, nxtsen in dataloader:
            if step > t_total:
                lg.log("Reached step limit, training done.")
                break
            inputs = inputs.to(device)
            ttype = ttype.to(device)
            mask = mask.to(device)
            lm = lm.to(device)
            nxtsen = nxtsen.to(device)
            if method == 'both':
                loss = model(inputs, ttype, mask, lm, nxtsen)
            elif method == 'masklm':
                loss = model(inputs, ttype, mask, lm)
            else:
                raise ValueError("How do you come to this place???")
            if (batch_num + 1) % 50 == 0 or (batch_num + 1) == t_batch:
                lg.log("epoch {}/{}, batch {}/{}, loss = {:.6f}".format(epoch + 1, t_epoch, batch_num + 1, t_batch,
                                                                        loss.item()))
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_num += 1
            step += 1

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
                   read_from_cache=False, language='english'):
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
    elif model_name == "transformer":
        model = TransformerClassifier(num_class).to(device)
        lg.log("choosing Transformer model.")
    else:
        raise ValueError("No such model named {}.".format(model_name))
    model.train()

    init_epoch = 0
    epoch_dict = {'textrnn': 10, 'textcnn': 30, 'transformer': 10}
    lr_dict = {'textrnn': 1e-3, 'textcnn': 1e-5, 'transformer': 1e-4}
    t_epoch = epoch_dict[model_name]
    lr = lr_dict[model_name]
    criterion = nn.CrossEntropyLoss()
    if model_name in ["textrnn", "textcnn"]:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = BertAdam(model.parameters(), lr=lr, warmup=0.04, b2=0.9998, e=1e-9, t_total=t_batch * t_epoch)
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
            # output = model(inputs) if model_name == "textcnn" else output = model(inputs, length)
            if model_name in ["textcnn", "transformer"]:
                output = model(inputs)
            elif model_name == "textrnn":
                output = model(inputs, length)
            else:
                raise SystemError("Oh!")
            # N * output_size (after softmax, represent probability)  eg. N * 2
            loss = criterion(output, label)
            if (batch_num + 1) % 50 == 0 or (batch_num + 1) == t_batch:
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
                further_pretrained=None, state_path=None, read_from_cache=False, language="english"):
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
        model = SimpleBert(512, num_class, language=language).to(device)
        lg.log("choosing BERT + Linear model.")
    elif model_name == "bert_lstm":
        model = RecBert(512, 1024, num_class, language=language).to(device)
        lg.log("choosing BERT + {}LSTM model + final hidden state.".format("bi-directional "))
    elif model_name == "bert_lstm2":
        model = RecBert(512, 1024, num_class, language=language, method=2).to(device)
        lg.log("choosing BERT + {}LSTM model + all hidden state.".format("bi-directional "))
    elif model_name == "bert_lstm3":
        model = RecBert(512, 768, num_class, language=language, method=3).to(device)
        lg.log("choosing BERT + {}LSTM model + final hidden state + add&norm.".format("bi-directional "))
    else:
        raise ValueError("How???")
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
        for inputs, mask, label, length in trainloader:
            inputs = inputs.to(device)
            mask = mask.to(device)
            label = label.to(device)
            length = length.to(device)
            output = model(inputs, mask) if model_name == 'bert_linear' else model(inputs, mask, length)
            # N * output_size (after softmax, represent probability)  eg. N * 2
            loss = criterion(output, label)
            if (batch_num + 1) % 50 == 0 or (batch_num + 1) == t_batch:
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


def evaluate(task_name, model_path, datasets="IMDB", batch_size=24, model_name="linear",
             read_from_cache=False, language="english"):
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
        model = SimpleBert(512, num_class, language=language).to(device)
        lg.log("choosing BERT + Linear model.")
    elif model_name == "bert_lstm":
        model = RecBert(512, 1024, num_class, language=language).to(device)
        lg.log("choosing BERT + {}LSTM model + final hidden state.".format("bi-directional "))
    elif model_name == "bert_lstm2":
        model = RecBert(512, 1024, num_class, language=language, method=2).to(device)
        lg.log("choosing BERT + {}LSTM model + all hidden state.".format("bi-directional "))
    elif model_name == "bert_lstm3":
        model = RecBert(512, 768, num_class, language=language, method=3).to(device)
        lg.log("choosing BERT + {}LSTM model + final hidden state + add&norm.".format("bi-directional "))
    elif model_name == "textrnn":
        model = TextRNN(512, 1024, num_class).to(device)
        lg.log("choosing {}TextRNN model.".format("bi-directional "))
    elif model_name == "textcnn":
        model = TextCNN(512, 8, num_class, (5, 5)).to(device)
        lg.log("choosing TextCNN model.")
    elif model_name == "transformer":
        model = TransformerClassifier(num_class).to(device)
        lg.log("choosing transformer classifier.")
    else:
        raise ValueError("No such model named {}".format(model_name))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # evaluate
    lg.log("Testing......")
    val_loss = 0.0
    val_total = 0
    val_cor = 0
    tf_matrix = torch.zeros([num_class, num_class])
    label_count = torch.zeros([num_class])
    predict_count = torch.zeros([num_class])
    batch_num = 0
    with torch.no_grad():
        for inputs, mask, label, length in testloader:
            inputs = inputs.to(device)
            mask = mask.to(device)
            label = label.to(device)
            length = length.to(device)
            # output = model(inputs, mask) if model_name in ["bert_linear", "bert_lstm"] else \
            #     (model(inputs) if model_name == "textcnn" else model(inputs, length))
            if model_name in bert_dict:
                output = model(inputs, mask) if model_name == 'bert_linear' else model(inputs, mask, length)
            elif model_name in ["textcnn", "transformer"]:
                output = model(inputs)
            elif model_name == "textrnn":
                output = model(inputs, length)
            else:
                raise SystemError("Oh")
            # N * output_size (after softmax, represent probability)  eg. N * 2
            loss = criterion(output, label)
            val_loss += loss.item()

            prediction = output.argmax(dim=-1)
            answer = label.view(-1)
            val_total += prediction.shape[0]
            val_cor += prediction[prediction == answer].shape[0]
            for i in range(prediction.shape[0]):
                tf_matrix[answer[i]][prediction[i]] += 1
                label_count[answer[i]] += 1
                predict_count[answer[i]] += 1
            if (batch_num + 1) % 50 == 0 or (batch_num + 1) == t_batch:
                lg.log("Testing {} / {} done.".format(batch_num + 1, t_batch))
            batch_num += 1

    val_loss = val_loss / t_batch
    acc = val_cor / val_total
    macro_f1, micro_f1 = f1_count(tf_matrix, label_count, predict_count, lg)
    lg.log("Test Result: {} / {} correct, {} accuracy, {} average loss, {} macro_f1, {} micro_f1".format(val_cor,
                                                                                                         val_total, acc,
                                                                                                         val_loss,
                                                                                                         macro_f1,
                                                                                                         micro_f1),
           message=push_message)

    lg.writelog()


# ======================TRAINING SCRIPTS=========================


def valid(args):
    if args.data not in dataset_dict:
        return 2, "Dataset not found ({} is not in the dataset dict)".format(args.data)
    if args.model not in model_dict:
        return 2, "Model not found ({} is not in the model dict)".format(args.model)
    if args.further_pretraining and args.model not in bert_dict:
        return 2, "Further pretraining can only perform in Bert Models."
    if args.fine_tuning and args.model not in bert_dict:
        return 2, "Fine tuning can only perform in Bert Models."
    if args.training and args.model in bert_dict:
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
    print("Dataset: {}".format(args.data))
    print("    number of class: {}".format(dataset_dict[args.data]['num_class']))
    print("    language: {}".format(dataset_dict[args.data]['language']))
    print("    further_prertaining method: {}".format(dataset_dict[args.data]['ftp_method']))
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
        phase += 1
    if args.shut:
        print("Phase {}: Shut Down.".format(phase))
        print("    Shut down the server no matter it runs smoothly or not.")
        phase += 1

def session(args):
    if args.alarm:
        global push_message
        push_message = True
    if args.debug:
        datasets.debugging = True
    if args.further_pretraining:
        task_name = "{}_FtP".format(args.name)
        further_pretraining(task_name, args.data, args.ftp_batch_size, args.ftp_state_path, args.read_from_cache,
                            dataset_dict[args.data]['language'], dataset_dict[args.data]['ftp_method'])
    if args.fine_tuning:
        if args.further_pretraining:
            if args.fit_ftp_path:
                ftp_path = args.fit_ftp_path
            else:
                ftp_path = "/root/autodl-nas/checkpoint/{}_FtP.pb".format(args.name)
        else:
            ftp_path = args.fit_ftp_path
        fine_tuning(args.name, args.data, args.fit_batch_size, args.model, ftp_path, args.fit_state_path,
                    args.read_from_cache, dataset_dict[args.data]['language'])
    if args.training:
        basis_training(args.name, args.data, args.train_batch_size, args.model, args.train_state_path,
                       args.read_from_cache, dataset_dict[args.data]['language'])
    if args.testing:
        if args.fine_tuning or args.training:
            if args.test_model_path:
                model_path = args.test_model_path
            else:
                model_path = "/root/autodl-nas/checkpoint/{}.pb".format(args.name)
        else:
            model_path = args.test_model_path
        evaluate(args.name, model_path, args.data, args.test_batch_size, args.model,
                 args.read_from_cache, dataset_dict[args.data]['language'])


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
    parser.add_argument("--shut", help="Shut down server after session is terminated (smoothly or interrupted)",
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
    print("Session terminated smoothly.")
    if args.shut:
        print("Server will be shut down in 10 seconds......")
        time.sleep(10)
        os.system("shutdown")
