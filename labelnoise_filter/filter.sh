#!/bin/bash
#

KEY=6
SPEED=200
CUDA_VISIBLE_DEVICES=0 python evaluate_locally.py bass ${KEY} ${SPEED} &
CUDA_VISIBLE_DEVICES=1 python evaluate_locally.py drums ${KEY} ${SPEED} &
CUDA_VISIBLE_DEVICES=2 python evaluate_locally.py vocals ${KEY} ${SPEED} &
CUDA_VISIBLE_DEVICES=3 python evaluate_locally.py other ${KEY} ${SPEED} &