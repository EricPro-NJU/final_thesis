#!/bin/sh

python running.py -n all_layer_mean -d IMDB -m bert_linear1 -fit -ev --fit_batch_size 20 --fit_ftp_path /root/autodl-nas/checkpoint/IMDB_FtP.pb --read_from_cache --alarm

