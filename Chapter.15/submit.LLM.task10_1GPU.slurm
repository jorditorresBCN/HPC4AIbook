#!/bin/bash

#SBATCH --job-name LLM_task10_1GPU 
#SBATCH --chdir . 
#SBATCH --output ./results/R-%x.%j.out 
#SBATCH --error ./results/R-%x.%j.err 

#SBATCH --nodes 1                   
#SBATCH --ntasks-per-node 1         
#SBATCH --gres gpu:1                
#SBATCH --cpus-per-task 80          
#SBATCH --time 00:29:59            
#SBATCH --account bsc31
#SBATCH --qos acc_debug
#SBATCH --exclusive


#### Variable to be modified by the student
GPUS_PER_NODE=1  # between 1 and 4 GPUs
MICRO_BATCH_SIZE=37
MODEL_PRECISION="bf16" # bf16 - fp32
MIXED_PRECISION="bf16" # bf16 - fp8 - no
ATTN=sdpa # eager - flash_attention_2 - sdpa
LIGER_KERNEL=true # false - true 

module load singularity

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank \$SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

MODEL=LLAMA3.2-1B
PATH_TO_MODEL=Llama-3.2-1B

MAX_STEPS=450
SEQUENCE_LEN=1024
OPTIMIZER=adamw_torch
TORCH_COMPILE=false 

RUN_NAME="NODES-$SLURM_NNODES-GPUs-$GPUS_PER_NODE-$MODEL-MODEL-PRECISION-$MODEL_PRECISION-MIXED-PRECISION-$MIXED_PRECISION-ATTN-$ATTN-$OPTIMIZER-TC-$TORCH_COMPILE-LIGER-$LIGER_KERNEL-SEQLEN-$SEQUENCE_LEN-MBS-mak$MICRO_BATCH_SIZE-$(cat /proc/sys/kernel/random/uuid)"

PYTHON_FILE=./benchmark.py
PYTHON_ARGS="--path_to_model $PATH_TO_MODEL \
    --run_name $RUN_NAME \
    --max_steps $MAX_STEPS \
    --sequence_length $SEQUENCE_LEN \
    --per_device_train_batch_size $MICRO_BATCH_SIZE \
    --model_precision $MODEL_PRECISION \
    --attn $ATTN \
    --torch_compile $TORCH_COMPILE \
    --use_liger_kernel $LIGER_KERNEL \
    --optim $OPTIMIZER \
    --output_dir ./results/output \
    --save_strategy no \
    --report_to none \
     "


export CMD="ACCELERATE_MIXED_PRECISION=$MIXED_PRECISION $LAUNCHER $PYTHON_FILE $PYTHON_ARGS"

SINGULARITY_CONTAINER=/gpfs/apps/MN5/ACC/SINGULARITY/SRC/images/nvidiaPytorch24.07_bis.sif

SINGULARITY_ARGS=" \
    --nv \
    $SINGULARITY_CONTAINER \
    "

SRUN_ARGS=" \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --jobid $SLURM_JOB_ID \
    "

echo "$CMD"

srun $SRUN_ARGS bsc_singularity exec  $SINGULARITY_ARGS bash -c "$CMD"
