#!/bin/bash

#SBATCH --partition=<FILL ME IN!>
#SBATCH --job-name=<FILL ME IN!>
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --comment=<FILL ME IN!>
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem 100G
#SBATCH --array=0-<FILL ME IN!>%<FILL ME IN!>
#SBATCH --requeue

echo "Processing job $SLURM_ARRAY_TASK_ID.yml"

cd <FILL ME IN!>

FILE=logs/
if test -d "./$FILE"; then
    echo ""
else
    echo "$FILE does not exist in processing dir."
    exit 1
fi

export PATH="/admin/home-$USER/miniconda3/condabin:$PATH" # export conda in path <FILL ME IN!>
source /admin/home-$USER/miniconda3/etc/profile.d/conda.sh # source conda <FILL ME IN!>

conda activate <FILL ME IN!> # your conda enironment with dataset2metadata

srun python dataset2metadata --yml ./jobs/$SLURM_ARRAY_TASK_ID.yml
