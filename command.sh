#!/bin/sh

python running.py -n Sogou_BLN_fit -d Sogou -m bert_linear -fit -ev --read_from_cache
python running.py -n Sogou_BRNN_fit -d Sogou -m bert_lstm -fit -ev --read_from_cache

shutdown