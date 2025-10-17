#!/bin/bash
#SBATCH --job-name=jacobi-scalability
#SBATCH -o %x_%J.out
#SBATCH -e %x_%J.err

#SBATCH --nodes=
#SBATCH --time=
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
##SBATCH --gpu-bind=none
#SBATCH --exclusive

#SBATCH --account=
#SBATCH --partition=acc

#SBATCH --qos=

module purge
module load nvidia-hpc-sdk/23.11-cuda11.8
make clean
make cub

sizes=(32768 16384 8192)
tasks=( )

for size in "${sizes[@]}"; do
  for np in "${tasks[@]}"; do
    mpirun -np $np ./jacobi -niter 100000 -nx $size -ny $size
  done
done
