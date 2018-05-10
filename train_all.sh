#!/usr/bin/env bash

python3 ./main.py --batch_size 20 --epochs 120 --class sand --train
python3 ./train_infinite_generator.py
python3 ./main.py --batch_size 20 --epochs 120 --class roads --input_size 64 --train
python3 ./main.py --batch_size 20 --epochs 120 --class grass --train
python3 ./main.py --batch_size 20 --epochs 120 --class ground --train
