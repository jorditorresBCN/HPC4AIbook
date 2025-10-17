#!/bin/bash
#SBATCH --job-name=matrix_multiplication
#SBATCH -o %x_%J.out
#SBATCH -e %x_%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 00:15:00
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --account=
#SBATCH --qos=

module load CUDA

# Compile the program (not best practice in production, but fine for learning)
nvcc -o mamatrixMulN matrixMulN.cu

# Execute the program
nsys nvprof ./matrixMulN
