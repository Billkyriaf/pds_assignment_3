#!/bin/bash
#SBATCH --job-name=PDS_SEQ
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00

#nvidia-smi
./build/single_block.out

 