#!/bin/sh

python running.py -n Sogou_textrnn -d Sogou -m textrnn  -ev --read_from_cache --test_model_path /root/autodl-nas/checkpoint/Sogou_textrnn.pb
python running.py -n Sogou_textcnn -d Sogou -m textcnn -tr -ev --read_from_cache
python running.py -n Sogou_trans -d Sogou -m transformer -tr -ev --read_from_cache

shutdown