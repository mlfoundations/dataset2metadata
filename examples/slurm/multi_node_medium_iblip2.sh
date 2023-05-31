#!/bin/bash

#SBATCH --partition=g40
#SBATCH --job-name=benchmark_d2m
#SBATCH --output=/admin/home-sy/repos/dataset2metadata/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --comment=datanet
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem 100G
#SBATCH --array=0-802%96
#SBATCH --requeue

echo "Processing job $SLURM_ARRAY_TASK_ID.yml"

cd /admin/home-sy/repos/dataset2metadata

FILE=logs/
if test -d "./$FILE"; then
    echo ""
else
    echo "$FILE does not exist in processing dir."
    exit 1
fi

export PATH="/admin/home-$USER/miniconda3/condabin:$PATH"
source /admin/home-$USER/miniconda3/etc/profile.d/conda.sh

conda activate d2m_tests
export PYTHONPATH=.

srun python dataset2metadata/main.py --yml ./jobs/$SLURM_ARRAY_TASK_ID.yml
