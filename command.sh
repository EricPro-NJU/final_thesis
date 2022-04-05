#!/bin/sh

python running.py -n Sogou_textrnn -d Sogou -m textrnn -tr -ev --read_from_cache
python running.py -n Sogou_textcnn -d Sogou -m textcnn -tr -ev --read_from_cache
python running.py -n Sogou_trans -d Sogou -m transformer -tr -ev --read_from_cache

shutdown