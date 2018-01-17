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
voltash
module load cuda/9.1
module load gcc/6.3.0

matsize=1000
max_iter = 1000

./program_jac_mp_v3 $matsize $matsize $matsize max_iter

## data dirs
#mkdir -rp analysis/poisson
#rm -rf analysis/poisson/*

#echo "\n ... \n"

#er_print -func test.blk.opt.1.er > test.blk.opt.1.txt

#echo "\n ... \n"
