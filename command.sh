#!/bin/sh

python running.py -n IMDB_BRNN_addnorm_FIT -d IMDB -m bert_lstm3 -fit -ev --read_from_cache
# python running.py -n IMDB_BRNN_final_FPT100kFIT -d IMDB -m bert_lstm -fit -ev --fit_ftp_path /root/autodl-nas/checkpoint/IMDB_FtP.pb --read_from_cache
shutdown