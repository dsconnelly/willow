#!/bin/bash -e

#SBATCH --tasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --out=/scratch/dsc7746/willow/slurm.out
#SBATCH --mail-user=dsconnelly@nyu.edu
#SBATCH --mail-type=END,FAIL

cd /scratch/dsc7746/willow

args="data/control-1e7 models/${SLURM_JOB_NAME}"
singularity exec --overlay ${overlay}:ro ${image} /bin/bash -c \
    "source /ext3/activate.sh; python -um willow train-forest ${args}"
