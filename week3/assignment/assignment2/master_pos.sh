#!/bin/bash

filename='project3pos'

## cluster setup
#BSUB -J project3pos
#BSUB -o project3pos%J.out
#BSUB -q gpuv100
#BSUB -gpu "num=2:mode=exclusive_process:mps=yes?
## set wall time hh:mm
#BSUB -W 00:20 
#BSUB -R "rusage[mem=4096MB] span[hosts=1]"
## set number of cores
#BSUB -n 12

##  load modules
module load cuda/9.1
module load gcc/6.3.0

## data dirs
mkdir -rp analysis/poisson
rm -rf analysis/poisson/*
