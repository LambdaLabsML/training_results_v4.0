#!/bin/bash

cd ../pytorch
source config_G593-SD1_001x08x032.sh
export CONT=mlperf_trainingv4.0-gigacomputing:retinanet
export DATADIR=/path/to/datasets
export LOGDIR=/path/to/folder
export BACKBONE_DIR=/path/to/torch-home
./run_with_docker.sh
