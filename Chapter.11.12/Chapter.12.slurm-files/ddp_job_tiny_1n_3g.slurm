#!/bin/bash
#SBATCH --job-name=ddp_tiny_1nodes_3gpus
#SBATCH --chdir=.
#SBATCH --output=./results/R-%x.%j.out
#SBATCH --error=./results/R-%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=60
#SBATCH --time=00:30:00
#SBATCH --account=bsc31
#SBATCH --qos=acc_debug
#SBATCH --exclusive


module purge
module load singularity

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

MODEL="vit"
DATASET="./datasets/tiny-224"
EPOCHS=5
BS=128
NW=10
OPTIM="sgd"

LAUNCHER="torchrun \
    --nproc_per_node=3 \
    --nnodes=1 \
    --node_rank=$SLURM_PROCID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend=c10d"

PYTHON_FILE="./train_ddp.py"
PYTHON_ARGS="--model_name $MODEL \
             --dataset $DATASET \
             --num_epochs $EPOCHS \
             --batch_size $BS \
             --eval_batch_size $BS \
             --num_workers $NW \
             --optimizer $OPTIM \
             --mixed_precision bf16 \
             --compile"

export CMD="$LAUNCHER $PYTHON_FILE $PYTHON_ARGS"

SINGULARITY_CONTAINER="/gpfs/apps/MN5/ACC/SINGULARITY/SRC/images/nvidiaPytorch24.07"
SINGULARITY_ARGS="--nv $SINGULARITY_CONTAINER"
SRUN_ARGS="--cpus-per-task $SLURM_CPUS_PER_TASK --jobid $SLURM_JOB_ID"

srun $SRUN_ARGS singularity exec $SINGULARITY_ARGS bash -c "$CMD"

