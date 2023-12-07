#!/bin/bash

set -e

python () {
    cmd="python $@"
    singularity exec --overlay ${overlay}:ro ${image} /bin/bash -c \
        "source /ext3/activate.sh; ${cmd}"
}

data_dir="data/390"
control_dir="/scratch/dsc7746/cases/390"
warm_dir="/scratch/dsc7746/cases/800"
warmest_dir="/scratch/dsc7746/cases/1200"
suffix="z"

submit() {
    local model_name=$1
    local model_dir="models/${model_name}"
    
    mkdir -p ${model_dir}
    touch "${model_dir}/model.pkl"

    local train_cmd="train-emulator ${data_dir} ${model_dir}"
    local train_id=$(sbatch --parsable -J ${model_name}-train willow.slurm $train_cmd)

    local shap_cmd="save-shapley-values ${data_dir} ${model_dir}"
    sbatch -J ${model_name}-shapley --dependency=afterok:${train_id} willow.slurm ${shap_cmd}

    ~/MiMA/scripts/create_case "${control_dir}/${model_name}" 60 \
        "${control_dir}/spinup" "" 390 "${model_dir}/model.pkl"

    ~/MiMA/scripts/create_case "${warm_dir}/${model_name}" 60 \
        "${warm_dir}/spinup" "" 800 "${model_dir}/model.pkl"

    ~/MiMA/scripts/create_case "${warmest_dir}/${model_name}" 60 \
        "${warmest_dir}/spinup" "" 1200 "${model_dir}/model.pkl"

    sbatch -J ${model_name}-control --dependency=afterok:${train_id} \
    sbatch -J ${model_name}-control \
        ${control_dir}/${model_name}/submit.slurm

    sbatch -J ${model_name}-warmer --dependency=afterok:${train_id} \
    sbatch -J ${model_name}-warmer \
        ${warm_dir}/${model_name}/submit.slurm

    sbatch -J ${model_name}-warmest --dependency=afterok:${train_id} \
    sbatch -J ${model_name}-warmest \
        ${warmest_dir}/${model_name}/submit.slurm
}

for model_name in $(./names.sh); do
    submit "${model_name}-${suffix}"
done