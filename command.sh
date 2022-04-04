#!/bin/sh

python running.py -n Sogou_BLN_fit -d Sogou -m bert_linear -fit -ev --read_from_cache --fit_batch_size 20
python running.py -n Sogou_BRNN_fit -d Sogou -m bert_lstm -fit -ev --read_from_cache

shutdown