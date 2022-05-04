#!/bin/sh

python running.py -n all_layer_mean_revise -d IMDB -m bert_linear1 -fit -ev --fit_ftp_path /root/autodl-nas/checkpoint/IMDB_FtP.pb --read_from_cache --alarm

