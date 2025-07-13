#!/bin/bash

#SBATCH --job-name LLM_HelloWorld

#SBATCH --chdir .
#SBATCH --output ./results/R-%x.%j.out # local directory
#SBATCH --error ./results/R-%x.%j.err # local directory

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=80
#SBATCH --time=00:29:59

#SBATCH --account bsc31 
#SBATCH --qos acc_debug
#SBATCH --exclusive

module purge
module load singularity

# Execution of script
SRUN_ARGS=" \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --jobid $SLURM_JOB_ID \
    "

export WANDB_DISABLED="1"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
export ACCELERATE_MIXED_PRECISION="no"

MODEL_PATH="/gpfs/projects/bsc31/LABS/L6.cc-mei.LLM/LLM/facebook-opt-1.3b"
DATA_PATH="/gpfs/projects/bsc31/LABS/Lx-Inference/data"


srun $SRUN_ARGS bsc_singularity exec --nv /gpfs/apps/MN5/ACC/SINGULARITY/SRC/images/nvidiaPytorch24.07_bis.sif torchrun --nproc_per_node 1 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_backend c10d --max_restarts 0 --tee 3  HelloWorld.py --model_path $MODEL_PATH --data_path $DATA_PATH
