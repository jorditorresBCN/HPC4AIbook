#!/bin/bash
#SBATCH --job-name EX00.BechmarkModels

#SBATCH --chdir . 
#SBATCH --output ./results/R-%x.%j.out
#SBATCH --error ./results/R-%x.%j.err

#SBATCH --nodes 1                   
#SBATCH --ntasks-per-node 1         
#SBATCH --gres gpu:1                
#SBATCH --cpus-per-task 20          
#SBATCH --time 02:00:00      
#SBATCH --account bsc31
#SBATCH --qos acc_debug
#SBATCH --exclusive


module purge
module load singularity

export CMD="python3 ./benchmark_models.py" # models custom & vit hardcoded

SINGULARITY_CONTAINER=/gpfs/apps/MN5/ACC/SINGULARITY/SRC/images/nvidiaPytorch24.07
SINGULARITY_ARGS=" --nv  $SINGULARITY_CONTAINER"

SRUN_ARGS=" --cpus-per-task $SLURM_CPUS_PER_TASK --jobid $SLURM_JOB_ID"
srun $SRUN_ARGS bsc_singularity exec  $SINGULARITY_ARGS bash -c "$CMD"
