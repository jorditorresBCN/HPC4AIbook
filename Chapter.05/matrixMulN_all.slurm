#!/bin/bash
#SBATCH --job-name=matrixMulN_all
#SBATCH -o %x_%J.out
#SBATCH -e %x_%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 00:35:00
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --account=
#SBATCH --qos=

module purge
module load cuda

# Compile the program (not best practice in production, but fine for learning)
nvcc -o matrixMulN_all matrixMulN_all.cu

for N in 64 128 256 512 1024 2048 4096 8192 ; do
  echo "========================================="
  echo "Running matrix multiplication with N=$N"
  echo "========================================="
  nsys nvprof ./matrixMulN_all $N
done
