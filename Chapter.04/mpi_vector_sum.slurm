#!/bin/bash
#SBATCH --job-name=mpi_vector_sum
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --account=bsc31
#SBATCH --qos=gp_debug

module load oneapi/2023.2.0

mpiicx -O2 -o mpi_vector_sum mpi_vector_sum.c

srun ./mpi_vector_sum
