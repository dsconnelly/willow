#!/bin/bash

python () {
    cmd="python $@"
    singularity exec --overlay ${overlay}:ro ${image} /bin/bash -c \
        "source /ext3/activate.sh; ${cmd}"
}

data_dir="data/ad99-control"
plot_dir="plots/manuscript"

models=$(./names.sh models s wind-T-s ,)
controls=$(./names.sh ../cases/control s wind-T-s ,)
warmers=$(./names.sh ../cases/4xco2 s "[ot]-wind-T-s" ,)
forests=$(./names.sh models s mubofo ,)

echo -n "Plotting example AD99 input and output profiles ..."
python -m willow plot-example-profiles \
    ${data_dir} ${models} ${plot_dir}/drag-examples.png
echo " done."

echo -n "Plotting R2 scores for representative modes ..."
python -m willow plot-R2-scores \
    ${data_dir} ${models} ${plot_dir}/three-R2-scores.png
echo " done."

echo -n "Plotting R2 scores for boosted forest ..."
python -m willow plot-R2-scores \
    ${data_dir} ${forests} ${plot_dir}/mubofo-R2-scores.png
echo " done."

echo -n "Plotting Shapley and Gini importances for a boosted forest ..."
python -m willow plot-feature-importances \
    ${data_dir} models/mubofo-wind-T-s \
    ${plot_dir}/mubofo-importances.png \
    --gini True
echo " done."

echo -n "Plotting Shapley importances for representative models ..."
python -m willow plot-feature-importances \
    ${data_dir} models/ad99-wind-T,${models} \
    ${plot_dir}/three-importances.png
echo " done."

echo -n "Plotting QBOs for representative models ..."
python -m willow plot-qbos \
    ${controls} ${plot_dir}/control-qbos.png
echo " done."

echo -n "Plotting SSWs for representative models ..."
python -m willow plot-ssws \
    ${controls} ${plot_dir}/control-ssws.png
echo " done."

echo -n "Plotting QBOs for representative models in 4xCO2 runs ..."
python -m willow plot-qbos \
    ${warmers} ${plot_dir}/4xco2-qbos.png
echo " done."

echo -n "Plotting SSWs for representative models in 4xCO2 runs ..."
python -m willow plot-ssws \
    ${warmers} ${plot_dir}/4xco2-ssws.png
echo " done."