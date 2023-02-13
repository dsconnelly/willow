#!/bin/bash

python () {
    cmd="python $@"
    singularity exec --overlay ${overlay}:ro ${image} /bin/bash -c \
        "source /ext3/activate.sh; ${cmd}"
}

data_dir="data/ad99-control"
control_dir="/scratch/dsc7746/cases/control"
warm_dir="/scratch/dsc7746/cases/4xco2"
suffix="y"

submit() {
    local model_name=$1
    local model_dir="models/${model_name}"

    local train_cmd="train-emulator ${data_dir} ${model_dir}"
    local train_id=$(sbatch --parsable -J ${model_name}-train willow.slurm $train_cmd)

    local shap_cmd="plot-feature-importances ${data_dir} ${model_dir} plots/manuscript/${model_name}-shapley.png"
    sbatch -J ${model_name}-shapley --dependency=afterok:${train_id} willow.slurm ${shap_cmd}

    python -m willow initialize-coupled-run ${control_dir}/ad99 ${model_dir} --n-years 30
    sbatch --dependency=afterok:${train_id} ${control_dir}/${model_name}/submit.slurm

    python -m willow initialize-coupled-run ${warm_dir}/ad99 ${model_dir} --n-years 30
    sbatch --dependency=afterok:${train_id} ${warm_dir}/${model_name}/submit.slurm
}

model_names=$(./names.sh)
for model_name in "${model_names[@]}"; do
    submit "${model_name}-${suffix}"
done

