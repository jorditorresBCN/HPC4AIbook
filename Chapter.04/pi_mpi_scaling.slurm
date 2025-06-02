#!/bin/bash
#SBATCH --job-name=pi_scaling
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --ntasks=64     
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


for P in 2 4 8 16 32 64
do
  echo ""
  echo "=== RUNNING WITH $P PROCESSES ==="
  srun --ntasks=$P ./pi_mpi_timed
done
