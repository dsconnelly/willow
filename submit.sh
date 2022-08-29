#!/bin/bash

python () {
    cmd="python $@"
    singularity exec --overlay ${overlay}:ro ${image} /bin/bash -c \
        "source /ext3/activate.sh; ${cmd}"
}

suffix=$1
ddir="data/control-1e7"
cdir="/scratch/dsc7746/cases"

kinds=(
    mubofo
    xgboost
    random
    WaveNet
)

models=(
    mubofo-wind
    mubofo-wind-T
    mubofo-wind-Nsq
    mubofo-wind-shear
    mubofo-shear-Nsq
    mubofo-wind-Nsq-pca

    xgboost-wind-Nsq
    xgboost-wind-shear

    random-wind-Nsq
    random-wind-shear

    WaveNet-wind
    WaveNet-wind-T
    WaveNet-wind-Nsq
)

case $2 in

    "train")
        for model in "${models[@]}"; do
            name="${model}-${suffix}"
            args="train-emulator ${ddir} models/${name}"
            sbatch -J "${name}-train" willow.slurm ${args}
        done
        ;;

    "plot-offline")
        for kind in "${kinds[@]}"; do
            use=""
            for model in "${models[@]}"; do
                if [[ $model == ${kind}-* ]]; then
                    use="${use}models/${model}-${suffix},"
                fi
            done

            fname="plots/offline-scores/${kind}-${suffix}-scores.png"
            args="${ddir} ${use%?} ${fname}"
            python -m willow plot-offline-scores ${args}
        done
        ;;

    "plot-shapley")
        odir="plots/shapley"
        for model in "${models[@]}"; do
            name="${model}-${suffix}"
            cmd="plot-shapley-values"
            
            args="${cmd} models/${name} ${odir} --data-dir ${ddir}"
            sbatch -J "${name}-shapley" willow.slurm ${args}
        done
        ;;

    "online")
        for model in "${models[@]}"; do
            name="${model}-${suffix}"
            args="setup-mima ${cdir}/control models/${name}"
            python -m willow ${args}
            sbatch "${cdir}/${name}/submit.slurm"
        done
        ;;

    "plot-profiling")
        use="${cdir}/profiling"
        for model in "${models[@]}"; do
            use="${use},${cdir}/${model}-${suffix}"
        done

        args="${use} plots/profiling/profiling-${suffix}.png"
        python -m willow plot-online-profiling ${args}
        ;;

    "plot-qbos")
        for kind in "${kinds[@]}"; do
            use="${cdir}/control"
            for model in "${models[@]}"; do
                if [[ $model == ${kind}-* ]]; then
                    use="${use},${cdir}/${model}-${suffix}"
                fi
            done

            args="${use} plots/qbos/${kind}-qbos-${suffix}.png"
            python -m willow plot-qbos ${args}
        done
        ;;

    *)
        echo "unknown command $2"
        ;;
        
esac
