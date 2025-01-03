#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

TASK_NAME="models/exp32"
PICTURE_PATH="data/train"
METADATA_PATH="data/train.csv"
NUM_WORKERS=4
BATCH_SIZE_TRAIN=64
BATCH_SIZE_TEST=32
N_EPOCHS=40
LEARNING_RATE=0.001
MOMENTUM=0.9
NETWORK="pretrained_inceptionv4"
WARMUP="True"
N_FOLD=10


python train_k_fold.py \
    --task_name $TASK_NAME \
    --picture_path $PICTURE_PATH \
    --metadata_path $METADATA_PATH \
    --num_workers $NUM_WORKERS \
    --batch_size_train $BATCH_SIZE_TRAIN \
    --batch_size_test $BATCH_SIZE_TEST \
    --n_epochs $N_EPOCHS \
    --lr $LEARNING_RATE \
    --momentum $MOMENTUM \
    --network_file $NETWORK \
    --warmup $WARMUP \
    --n_fold $N_FOLD

