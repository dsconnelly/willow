#!/bin/bash -e

#SBATCH --tasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --out=/scratch/dsc7746/willow/slurm.out
#SBATCH --mail-user=dsconnelly@nyu.edu
#SBATCH --mail-type=END,FAIL

cd /scratch/dsc7746/willow
command="source /ext3/activate.sh; python -um willow $@"
singularity exec --overlay ${overlay}:ro ${image} /bin/bash -c "${command}"
