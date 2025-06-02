#!/bin/bash
#SBATCH --job-name=pi_compare
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --account=bsc31
#SBATCH --qos=gp_debug

module load oneapi/2023.2.0

echo "=== COMPILING ==="
icx -O3 pi_seq_timed.c -o pi_seq_timed
mpiicx -O3 pi_mpi_timed.c -o pi_mpi_timed

echo "=== RUNNING SEQUENTIAL ==="
./pi_seq_timed

echo "=== RUNNING PARALLEL ==="
srun ./pi_mpi_timed
