# final_thesis
## Research in Short Text Classification Based on Pre-trained Model BERT
This is a repo with my undergraduate thesis codes.

My thesis topic is BERT and news classification.

The codes run in ~~Google Colab~~ AutoDL.

Google Colab is too unstable. TAT

## Goal: Before 2020.4.1
1. Rewrite datasets.py

    ~~We notice that all the datasets that are required are in package torchtext. 
   We can write a general method/class to process the datasets.
   By indicating the datasets name, we can create different dataloaders
   in running scripts. In that case, running.py should be 
   rewritten too.~~
   
   F**K the internet error
   
2. Run scripts to get results.

    Use autodl server to get result of classification onfour datasets:
    1. IMDB
    2. Yelp
    3. AG news
    4. Sogou news
    
    By feeding into the following models:
    1. TextRNN
    2. TextCNN
    3. Transformer Encoder
    4. Bert+Linear
    5. Bert+RNN

## Abstract

In the era of big data, short text such as news and comments are growing significantly through the Internet. It is thus important to design topics or sentiment classification models to automatically identify valuable information. Traditional text classification methods have problems regarding feature extraction and model structures, such as sparsity of features, or lack of semantic relations. Pre-trained BERT model provides an end-to-end paradigm. It consists of 12 bi-directional Transformer structures, and can achieve high classification performance after being pre-trained in large corpus and fine-tuned.
    
This paper study short text classification with pre-trained model BERT. There are usually three phases regarding text classification with BERT, including Pre-processing, Further Pre-training and Fine-tuning. We make an improvement to the algorithm of Further Pre-training, especially addressing the problem of class imbalance. We compare classification performances of BERT and other baseline models and proved the exceeding performances of BERT model. Moreover, we observe and analyze the output of each layer of BERT model. By conducting experiments, We verify that outputs of lower layers focus more on low-level features like syntax, while those of higher layers focus more on high-level features like semantics. Weights of higher layer outputs tend to be greater when we make those weights learnable.Finally, we focus on the condition where the number of samples in training set is low. BERT can weaken the disadvantages of low quantity of data to a certain extent, and the Further Pre-training phase will contribute to higher performances. We analyze the pre-training mechanism as well as the training process, and verify the existence of such phenomenon.

## Full paper

click [here](https://tex.nju.edu.cn/read/brddygzjfpmr) to get to the paper repo.
