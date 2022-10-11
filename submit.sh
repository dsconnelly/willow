#!/bin/bash

python () {
    cmd="python $@"
    singularity exec --overlay ${overlay}:ro ${image} /bin/bash -c \
        "source /ext3/activate.sh; ${cmd}"
}

suffix=$1
ddir="data/ad99-control-1e7"
cdir="/scratch/dsc7746/cases"
perturb=""

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
    mubofo-wind-T-noloc

    xgboost-wind-T
    xgboost-wind-shear

    random-wind-T
    random-wind-shear

    WaveNet-wind
    WaveNet-wind-T
    WaveNet-wind-Nsq
    WaveNet-wind-T-noloc
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

            fname="plots/sparc-fall-22/scores/${kind}-${suffix}-scores.png"
            args="${ddir} ${use%?} ${fname}"
            python -m willow plot-offline-scores ${args}
        done
        ;;

    "plot-shapley")
        odir="plots/sparc-fall-22/shapley"
        for model in "${models[@]}"; do
            name="${model}-${suffix}"
            cmd="plot-feature-importances"
            
            args="${cmd} models/${name} ${odir} shapley --data-dir ${ddir}"
            sbatch -J "${name}-shapley" willow.slurm ${args}
        done
        ;;

    "online")
        for model in "${models[@]}"; do
            name="${model}-${suffix}"
            newcase="${cdir}/${name}-${perturb}"
            
            args="setup-mima ${cdir}/ad99-${perturb} models/${name}"
            python -m willow ${args} --case-dir ${newcase}
            sbatch "${newcase}/submit.slurm"
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
            use="${cdir}/ad99-${perturb}"
            for model in "${models[@]}"; do
                if [[ $model == ${kind}-* ]]; then
                    use="${use},${cdir}/${model}-${suffix}-${perturb}"
                fi
            done

            args="${use} plots/sparc-fall-22/qbos/${kind}-qbos-${suffix}-${perturb}.png"
            python -m willow plot-qbos ${args}
        done
        ;;

    "plot-ssws")
        for kind in "${kinds[@]}"; do
            use="${cdir}/ad99-${perturb}"
            for model in "${models[@]}"; do
                if [[ $model == ${kind}-* ]]; then
                    use="${use},${cdir}/${model}-${suffix}-${perturb}"
                fi
            done

            args="${use} plots/ssws/${kind}-ssws-${suffix}-${perturb}.png"
            python -m willow plot-ssws ${args}
        done
        ;;

    *)
        echo "unknown command $2"
        ;;
        
esac
