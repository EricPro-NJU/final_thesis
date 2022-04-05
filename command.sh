#!/bin/sh

python running.py -n Sogou_BLN_FTP100kFIT -d Sogou -m bert_linear -fit -ev --read_from_cache --fit_batch_size 20 --fit_ftp_path /root/autodl-nas/checkpoint/Sogou_FtP.pb --alarm

python running.py -n Sogou_BRNN_FTP100kFIT -d Sogou -m bert_lstm -fit -ev --read_from_cache --fit_batch_size 16 --fit_ftp_path /root/autodl-nas/checkpoint/Sogou_FtP.pb --alarm

shutdown