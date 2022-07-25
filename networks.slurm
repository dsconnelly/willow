#!/bin/bash -e

#SBATCH --tasks=1
#SBATCH --time=36:00:00
#SBATCH --mem=64GB
#SBATCH --out=/scratch/dsc7746/willow/models/%x/log.out
#SBATCH --mail-user=dsconnelly@nyu.edu
#SBATCH --mail-type=END,FAIL

cd /scratch/dsc7746/willow
singularity exec --overlay ${overlay}:ro ${image} /bin/bash -c \
    "source /ext3/activate.sh; python -um willow train-network data/control-10000000 models/${SLURM_JOB_NAME} WaveNet"
