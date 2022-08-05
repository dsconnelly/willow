#!/bin/bash

models=(
    mubofo-wind-T
    mubofo-wind-Nsq
    mubofo-wind-shear
    mubofo-shear-Nsq
    mubofo-wind-Nsq-pca

    xgboost-wind-Nsq
    xgboost-wind-shear

    random-wind-Nsq
    random-wind-shear

    WaveNet-wind-T
    WaveNet-wind-Nsq
)

python ()
{ 
    cmd="python $@";
    singularity exec --overlay ${overlay}:ro ${image} /bin/bash -c \
        "source /ext3/activate.sh; $cmd"
}

suffix="i"
data="data/control-1e7"

if [ $1 == "train" ] || [ $1 == "offline" ]; then
    for model in "${models[@]}"; do
        name="${model}-${suffix}"

        if [ $1 == "train" ]; then
            args="train-emulator ${data} models/${name}"
            sbatch -J "${name}-train" willow.slurm $args

        elif [ $1 == "online" ]; then
            args="setup-mima /scratch/dsc7746/cases/control models/${name}"
            python -m willow $args
            sbatch /scratch/dsc7746/cases/${name}/submit.slurm

        fi
    done

elif [ $1 == "plot-offline" ]; then
    kinds=(
        mubofo
        xgboost
        random
        WaveNet
    )

    for kind in "${kinds[@]}"; do
        args=""
        for model in "${models[@]}"; do
            if [[ $model == ${kind}-* ]]; then
                args="${args}models/${model}-${suffix},"
            fi
        done

        args="${data} ${args%?} plots/offline/${kind}-scores.png"
        python -m willow plot-offline-scores $args
    done

else
    echo "Unknown command $1"
fi