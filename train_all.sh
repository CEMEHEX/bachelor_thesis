#!/usr/bin/env bash

python3 ./train_infinite_generator.py --classic

python3 ./main.py --batch_size 20 --epochs 120 --class grass --train
python3 ./main.py --batch_size 20 --epochs 120 --class ground --train

python3 ./main.py --batch_size 20 --epochs 120 --class water --classic --train
python3 ./main.py --batch_size 20 --epochs 120 --class forest --classic --train
python3 ./main.py --batch_size 20 --epochs 120 --class sand --classic --train
python3 ./main.py --batch_size 20 --epochs 120 --class clouds --classic --train
python3 ./main.py --batch_size 20 --epochs 120 --class water --classic --train
python3 ./main.py --batch_size 20 --epochs 120 --class roads --input_size 64 --classic --train
python3 ./main.py --batch_size 20 --epochs 120 --class buildings --input_size 64 --classic --train
python3 ./main.py --batch_size 20 --epochs 120 --class grass --classic --train
python3 ./main.py --batch_size 20 --epochs 120 --class ground --classic --train