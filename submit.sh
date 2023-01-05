#!/bin/bash

python () {
    cmd="python $@"
    singularity exec --overlay ${overlay}:ro ${image} /bin/bash -c \
        "source /ext3/activate.sh; ${cmd}"
}

data_dir="data/ad99-control"
case_dir="/scratch/dsc7746/cases/control"

submit() {
    local model_name=$1
    local model_dir="models/${model_name}"

    local train_cmd="train-emulator ${data_dir} ${model_dir}"
    local train_id=$(sbatch --parsable -J ${model_name}-train willow.slurm $train_cmd)

    local shap_cmd="plot-feature-importances ${data_dir} ${model_dir} plots/manuscript/${model_name}-shapley.png"
    sbatch -J ${model_name}-shapley --dependency=afterok:${train_id} willow.slurm ${shap_cmd}

    python -m willow initialize-coupled-run ${case_dir}/ad99 ${model_dir}
    sbatch --dependency=afterok:${train_id} ${case_dir}/${model_name}/submit.slurm
}

model_names=(
    mubofo-wind
    mubofo-wind-T
    mubofo-wind-T-noloc
    mubofo-wind-shear
    mubofo-wind-T-n_estimators_10000-max_depth_2-max_features_0.9-max_samples_0.9

    random-wind-T
    random-wind-shear

    WaveNet-wind
    WaveNet-wind-T
    WaveNet-wind-noloc
)

for model_name in "${model_names[@]}"; do
    submit "${model_name}-${1}"
done