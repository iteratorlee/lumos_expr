#!/usr/bin/zsh

nohup python3 -u hpt.py -j 8 -i 0 -s 0 -e 5 >> logs/hpt_0.log &
nohup python3 -u hpt.py -j 8 -i 1 -s 5 -e 10 >> logs/hpt_1.log &
nohup python3 -u hpt.py -j 8 -i 2 -s 10 -e 15 >> logs/hpt_2.log &
nohup python3 -u hpt.py -j 8 -i 3 -s 15 -e 20 >> logs/hpt_3.log &
nohup python3 -u hpt.py -j 8 -i 4 -s 20 -e 25 >> logs/hpt_4.log &
nohup python3 -u hpt.py -j 8 -i 5 -s 25 -e 30 >> logs/hpt_5.log &
