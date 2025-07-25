#!/bin/bash

#SBATCH --job-name Ex06.JIT

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

MODEL=vit                # --model_name: custom | vit 

DS=./datasets/micro-224  # --dataset: tiny-224 | micro-224
EPOCHS=5                 # --num_epochs
NW=10                     # --num_workers
OPTIM=sgd                # --optimizer 
LOG_ITER=500             # --iteration_logging
EPOCHS_EVAL_FREQ=5       # --epochs_eval

BS=128                   # --batch_size & --eval_batch_size

    PYTHON_FILE=./train.py
    PYTHON_ARGS="--model_name $MODEL \
                 --dataset $DS \
                 --num_epochs $EPOCHS \
                 --batch_size $BS \
                 --eval_batch_size $BS \
                 --num_workers $NW \
                 --optimizer $OPTIM \
                 --iteration_logging $LOG_ITER \
                 --epochs_eval $EPOCHS_EVAL_FREQ \
                 --mixed_precision bf16 \
                 --compile \
                "

    export CMD="python3 $PYTHON_FILE $PYTHON_ARGS"

    SINGULARITY_CONTAINER=/gpfs/apps/MN5/ACC/SINGULARITY/SRC/images/nvidiaPytorch24.07
    SINGULARITY_ARGS=" --nv  $SINGULARITY_CONTAINER"

    SRUN_ARGS=" --cpus-per-task $SLURM_CPUS_PER_TASK --jobid $SLURM_JOB_ID"

    srun $SRUN_ARGS bsc_singularity exec  $SINGULARITY_ARGS bash -c "$CMD"

