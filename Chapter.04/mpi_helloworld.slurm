#!/bin/bash
#SBATCH -J mpirun_mpi_helloworld
#SBATCH -t 00:15
#SBATCH -o %x_%J.out
#SBATCH -e %xo_%J.err
# #SBATCH --nodes 1
#SBATCH --ntasks=4 
# #SBATCH --ntasks-per-node=4 
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --account 
#SBATCH --qos gp_debug

module load oneapi/2023.2.0

echo "mpirun ./mpi_helloworld:"
mpirun ./mpi_helloworld

echo "srun ./mpi_helloworld"
srun ./mpi_helloworld
