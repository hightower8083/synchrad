#!/usr/bin/env sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4 python compute_spectrum.py
