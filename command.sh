#!/bin/sh

python running.py -n IMDB_BRNN_REV1_FIT -d IMDB -m bert_lstm -fit -ev --read_from_cache --alarm
python running.py -n IMDB_BRNN_REV2_FIT -d IMDB -m bert_lstm2 -fit -ev --read_from_cache --alarm
python running.py -n IMDB_BRNN_REV1_FPT100kFIT -d IMDB -m bert_lstm -fit -ev --fit_ftp_path /root/autodl-nas/checkpoint/IMDB_FtP.pb --read_from_cache --alarm
python running.py -n IMDB_BRNN_REV2_FPT100kFIT -d IMDB -m bert_lstm2 -fit -ev --fit_ftp_path /root/autodl-nas/checkpoint/IMDB_FtP.pb --read_from_cache --alarm
shutdown