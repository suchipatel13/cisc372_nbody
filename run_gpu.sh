#!/bin/bash -l
#SBATCH --job-name=nbody_cuda
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --partition=gpu-v100
#SBATCH --output=run_output_%j.txt
#SBATCH --error=run_error_%j.txt

vpkg_require gcc
vpkg_require cuda

cd cisc372_nbody

make clean
make

srun ./nbody > results.txt

