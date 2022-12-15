#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
# e.g. The jobname below is "example"
#SBATCH --job-name=MAE

# Each job will utilise all of brainstorm's resources.
# Here, we ask for 1 node with exlusive
# brainstorm has 256 CPU cores and 8 A100 GPUs.
#SBATCH --mem=700gb                   # Job memory request
#SBATCH --time=7-00:00:00            #The walltime
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task
###SBATCH -n 140                        # Number of CPUs <- THIS IS INCORRECT
#SBATCH --cpus-per-task=140         # Number of CPU cores per task <-USE THIS INSTEAD
#SBATCH --gres=gpu:8                # Number of GPUs to allocate to this job

# This is where the actual work is done. In this case, the script only waits.
# The time command is optional, but it may give you a hint on how long the
# command worked

nvidia-smi
nvcc --version
source /home/cl522/miniconda3/etc/profile.d/conda.sh
conda activate mmself
cd /home/cl522/github_repo/mmselfsup/tools/

CONFIG="/home/cl522/github_repo/mmselfsup/configs/selfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_allCXR.py"

GPUS=8
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3} \
    --work-dir="/home/cl522/github_repo/mmselfsup/tools/MAE_allCXR"
