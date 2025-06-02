#!/bin/bash
#SBATCH --job-name=pi_mpi
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --exclusive
#SBATCH --account=bsc31
#SBATCH --qos=gp_debug

module load oneapi/2023.2.0
mpiicx pi_mpi.c -o pi_mpi
srun ./pi_mpi

