#!/bin/bash

python () {
    cmd="python $@"
    singularity exec --overlay ${overlay}:ro ${image} /bin/bash -c \
        "source /ext3/activate.sh; ${cmd}"
}

data_dir="data/ad99-control"
plot_dir="plots"

models=$(./names.sh models s wind-T-s ,)
controls=$(./names.sh ../cases/control s wind-T-s ,)
warmers=$(./names.sh ../cases/4xco2 s "[ot]-wind-T-s" ,)
forests=$(./names.sh models s mubofo ,)

# echo -n "Plotting example AD99 input and output profiles ..."
# python -m willow plot-example-profiles \
#     ${data_dir} ${models} ${plot_dir}/methodology/drag-examples.png
# echo " done."

# echo -n "Plotting R2 scores for representative models ..."
# python -m willow plot-R2-scores \
#     ${data_dir} ${models} ${plot_dir}/offline/three-R2-scores.png
# echo " done."

# echo -n "Plotting R2 scores for boosted forests ..."
# python -m willow plot-R2-scores \
#     ${data_dir} ${forests} ${plot_dir}/offline/mubofo-R2-scores.png
# echo " done."

# echo -n "Plotting Shapley and Gini importances for a boosted forest ..."
# python -m willow plot-feature-importances \
#     ${data_dir} models/mubofo-wind-T-s \
#     ${plot_dir}/offline/mubofo-importances.png \
#     --gini True
# echo " done."

# echo -n "Plotting Shapley importances for representative models ..."
# python -m willow plot-feature-importances \
#     ${data_dir} models/ad99-wind-T,${models} \
#     ${plot_dir}/offline/three-importances.png
# echo " done."

echo -n "Plotting Shapley analytics for representative models ..."
python -m willow plot-shapley-analytics \
    models/ad99-wind-T ${models} ${plot_dir}/offline/three-analytics.png
echo " done."

# echo -n "Plotting QBOs for representative models ..."
# python -m willow plot-qbos \
#     ${controls} ${plot_dir}/online/control-qbos.png
# echo " done."

# echo -n "Plotting SSWs for representative models ..."
# python -m willow plot-ssws \
#     ${controls} ${plot_dir}/online/control-ssws.png
# echo " done."

# echo -n "Plotting QBOs for representative models in 4xCO2 runs ..."
# python -m willow plot-qbos \
#     ${warmers} ${plot_dir}/online/4xco2-qbos.png
# echo " done."

# echo -n "Plotting SSWs for representative models in 4xCO2 runs ..."
# python -m willow plot-ssws \
#     ${warmers} ${plot_dir}/online/4xco2-ssws.png
# echo " done."