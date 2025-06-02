#!/bin/bash
#SBATCH --job-name=mpi_sendrecv
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --ntasks=8                # Total number of MPI tasks
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --exclusive
#SBATCH --account=bsc31
#SBATCH --qos=gp_debug

module load oneapi/2023.2.0
mpiicx -o mpi_sendrecv mpi_sendrecv.c

srun ./mpi_sendrecv
