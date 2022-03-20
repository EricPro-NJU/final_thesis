# final_thesis
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
