#!/bin/bash
#SBATCH -J mpi_helloworld_4x4
#SBATCH -t 00:15
#SBATCH -o %x_%J.out
#SBATCH -e %xo_%J.err
#SBATCH --ntasks=16            # Total number of MPI tasks
#SBATCH --nodes=4              # Number of nodes to allocate
#SBATCH --ntasks-per-node=4    # Number of tasks per node
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --account bsc31
#SBATCH --qos gp_debug

module load oneapi/2023.2.0
mpiicx -o mpi_helloworld4x4 mpi_helloworld4x4.c  
srun ./mpi_helloworld4x4
