#!/bin/bash
#SBATCH --chdir .
#SBATCH --job-name=ResNet50    
#SBATCH --output=%x.%j.out     
#SBATCH --error=%x.%j.err     
#SBATCH --nodes 1                   
#SBATCH --ntasks-per-node 1                                       
#SBATCH --gres gpu:4           
#SBATCH --cpus-per-task 80          
#SBATCH --time 01:15:00           
#SBATCH --account nct_XXX       
#SBATCH --qos acc_debug     

echo " "
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Submit Directory: $SLURM_SUBMIT_DIR"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Account: $SLURM_JOB_ACCOUNT"
echo " "
echo "Number of Nodes Allocated: $SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST)"
echo "Total number of tasks: $SLURM_NTASKS"
echo "Number of tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Number of CPU cores per task: $SLURM_CPUS_PER_TASK"
echo " "
echo "Specific GPUs allocated:"
nvidia-smi


module purge
module load singularity

SINGULARITY_CONTAINER=/gpfs/scratch/nct_325/MN5-NGC-TensorFlow-23.03.sif
singularity exec --nv $SINGULARITY_CONTAINER python ResNet50.py --epochs 2 --batch_size 2048 --n_gpus 1
singularity exec --nv $SINGULARITY_CONTAINER python ResNet50.py --epochs 2 --batch_size 4096 --n_gpus 2
singularity exec --nv $SINGULARITY_CONTAINER python ResNet50.py --epochs 2 --batch_size 8192 --n_gpus 4

