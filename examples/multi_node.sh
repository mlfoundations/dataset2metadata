#!/bin/bash

#SBATCH --partition=g40
#SBATCH --job-name=dataset2metadata
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --comment=datanet
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem 100G
#SBATCH --array=0-UPPERBOUND
#SBATCH --requeue

echo "Processing job $SLURM_ARRAY_TASK_ID.yml"

cd LOCATION_OF_UR_REPO

FILE=logs/
if test -f "$FILE"; then
    echo ""
else
    echo "$FILE does not exist."
    exit 1
fi

export PATH="/admin/home-$USER/miniconda3/condabin:$PATH"
source /admin/home-$USER/miniconda3/etc/profile.d/conda.sh

conda activate dataset2metadata
srun dataset2metadata --yml examples/jobs/$SLURM_ARRAY_TASK_ID.yml
