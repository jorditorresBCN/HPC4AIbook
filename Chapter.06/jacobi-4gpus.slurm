#!/bin/bash
#SBATCH --job-name=jacobi-4gpus
#SBATCH -o %x_%J.out
#SBATCH -e %x_%J.err

#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive

#SBATCH --account=
#SBATCH --qos=

#module load oneapi/2023.2.0
module load nvidia-hpc-sdk/23.11-cuda11.8

make clean
#make default
#make optimized
make cub

mpirun -np 4 ./jacobi -niter 100000 -nx 16384 -ny 16384
