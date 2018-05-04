#!/usr/bin/env bash

./main.py --batch_size 20 --epochs 120 --train_data data/forest_train --test_data data/forest_test --weights weights/forest.h5 --logs out/forest.csv --train
./main.py --batch_size 20 --epochs 120 --train_data data/sand_train --test_data data/sand_test --weights weights/sand.h5 --logs out/sand.csv --train
./main.py --batch_size 20 --epochs 120 --train_data data/clouds_train --test_data data/clouds_test --weights weights/clouds.h5 --logs out/clouds.csv --train
./main.py --batch_size 20 --epochs 120 --train_data data/ground_train --test_data data/ground_test --weights weights/ground.h5 --logs out/ground.csv --train
./main.py --batch_size 20 --epochs 120 --train_data data/grass_train --test_data data/grass_test --weights weights/grass.h5 --logs out/grass.csv --train