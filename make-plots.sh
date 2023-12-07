#!/bin/bash
set -e

python () {
    cmd="python $@"
    singularity exec --overlay ${overlay}:ro ${image} /bin/bash -c \
        "source /ext3/activate.sh; ${cmd}"
}

plot() {
    echo -n "Plotting $1 ..."
    python -m willow "${@:2}"
    echo " done."
}

suffix="z"
data_dir="data/390"
plot_dir="plots"

models=$(./names.sh models ${suffix} wind-T-${suffix} ,)
repeats=$(./names.sh models ${suffix} [ot]-wind-T-[r${suffix}] ,)
forests=$(./names.sh models ${suffix} mubofo ,)

control_dir="390"
warmer_dir="800"
warmest_dir="1200"

controls=$(./names.sh ../cases/${control_dir} ${suffix} wind-T-${suffix} ,)
warmers=$(./names.sh ../cases/${warmer_dir} ${suffix} wind-T-${suffix} ,)
warmests=$(./names.sh ../cases/${warmest_dir} ${suffix} wind-T-${suffix} ,)

retrained="models/ad99-wind-T,models/mubofo-wind-T-z"
retrained="${retrained},models/mubofo-wind-T-lat_scale_1.55-z"

plot "example input and output profiles" \
    plot-example-profiles ${data_dir} ${models} \
    ${plot_dir}/offline/example-profiles.png

plot "R2 scores for representative models" \
    plot-R2-scores ${data_dir} ${models} \
    ${plot_dir}/offline/three-R2-scores.png \
    --n-samples 1000000

plot "R2 scores for boosted forests" \
    plot-R2-scores ${data_dir} ${forests} \
    ${plot_dir}/offline/mubofo-R2-scores.png \
    --n-samples 1000000

plot "R2 scores for representative models in ${warmest_dir}" \
    plot-R2-scores data/${warmest_dir} ${models} \
    ${plot_dir}/offline/three-R2-scores-${warmest_dir}.png \
    --n-samples 1000000

plot "SHAP and Gini importances for a boosted forest" \
    plot-feature-importances models/mubofo-wind-T-${suffix} \
    ${plot_dir}/offline/mubofo-importances.png \
    --gini True

plot "SHAP importances for representative models" \
    plot-feature-importances models/ad99-wind-T,${models} \
    ${plot_dir}/offline/three-importances.png

plot "SHAP importances for ${warmest_dir} samples" \
    plot-feature-importances models/ad99-wind-T,${models} \
    ${plot_dir}/offline/three-importances-${warmest_dir}.png \
    --suffix ${warmest_dir}

plot "SHAP error contributions" \
    plot-shapley-errors models/ad99-wind-T,${repeats} \
    ${plot_dir}/offline/shap-star.png

plot "SHAP importances of latitude for representative models" \
    plot-scalar-importances models/ad99-wind-T,${models} \
    ${plot_dir}/offline/shap-latitude.png

plot "SHAP importances of latitude with retrained forest" \
    plot-scalar-importances $retrained \
    ${plot_dir}/offline/shap-latitude-retrained-december.png

plot "biases in control climate" \
    plot-biases ${controls} \
    ${plot_dir}/online/control-biases.png

plot "QBOs in representative models" \
    plot-qbos ${controls},${warmers},${warmests} \
    ${plot_dir}/online/qbos.png

plot "QBO statistics for representative models" \
    plot-qbo-statistics ${controls},${warmers},${warmests} \
    ${plot_dir}/online/qbo-statistics.png

plot "SSW frequencies for representative models" \
    plot-ssw-frequencies ${controls},${warmers},${warmests} \
    ${plot_dir}/online/ssw-frequencies.png
