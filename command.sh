#!/bin/sh

python running.py -n Sogou_BLN_FTP100kFIT -d Sogou -m textrnn -tr -ev --read_from_cache
python running.py -n Sogou_BLN_FTP100kFIT -d Sogou -m textcnn -tr -ev --read_from_cache
python running.py -n Sogou_BLN_FTP100kFIT -d Sogou -m transformer -tr -ev --read_from_cache

shutdown