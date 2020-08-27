#!/bin/bash
set -xe

# data in ./data
train_data=lena
dev_data=moli

# train
python train.py --train_data ${train_data} --dev_data ${dev_data} --seq_len 40 --train_batch_size 16 --dev_batch_size 16
