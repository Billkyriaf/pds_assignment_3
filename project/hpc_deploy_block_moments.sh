#!/bin/bash
#SBATCH --job-name=PDS_SEQ
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00

nvidia-smi

nvprof ./build/block_moments.out
