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

control_dir="390"
warmer_dir="800"
warmest_dir="1200"

models=$(./names.sh models ${suffix} "" ,)
controls=$(./names.sh ../cases/${control_dir} ${suffix} "" ,)
warmers=$(./names.sh ../cases/${warmer_dir} ${suffix} "" ,)
warmests=$(./names.sh ../cases/${warmest_dir} ${suffix} "" ,)

names=(
    mubofo-wind-T-lat_scale_0.01-z
    mubofo-wind-T-lat_scale_0.85-z
    mubofo-wind-T-lat_scale_1.55-z
    mubofo-wind-T-lat_scale_2.05-z
)

# module load cdo/intel/1.9.10
# for name in ${names[@]}; do
#     cdo -O mergetime ~/scratch/cases/390/${name}/??/zonal_mean.nc ~/scratch/cases/390/${name}/midway_zonal_mean.nc
#     cdo -O mergetime ~/scratch/cases/800/${name}/??/zonal_mean.nc ~/scratch/cases/800/${name}/midway_zonal_mean.nc
#     cdo -O mergetime ~/scratch/cases/1200/${name}/??/zonal_mean.nc ~/scratch/cases/1200/${name}/midway_zonal_mean.nc
# done

# plot "R2 scores for forests with latitude scaling" \
#     plot-R2-scores ${data_dir} ${models} \
#     ${plot_dir}/lat-scale/three-R2-scores.png \
#     --n-samples 100000

# plot "R2 scores for forests with latitude scaling with 1200 ppm samples" \
#     plot-R2-scores data/1200 ${models} \
#     ${plot_dir}/lat-scale/three-R2-scores-1200.png \
#     --n-samples 100000

# plot "SHAP importances of latitude for forests with latitude scaling" \
#     plot-scalar-importances models/ad99-wind-T,${models} \
#     ${plot_dir}/lat-scale/latitude-shap.png

# plot "QBOs in forests with latitude scaling" \
#     plot-qbos ${controls},${warmers},${warmests} \
#     ${plot_dir}/online/retrained-qbos.png

plot "QBO statistics for forests with latitude scaling" \
    plot-qbo-statistics ${controls},${warmers},${warmests} \
    ${plot_dir}/online/retrained-qbo-statistics-december.png

# for name in ${names[@]}; do
#     rm -f ~/scratch/cases/390/${name}/??/midway_zonal_mean.nc
#     rm -f ~/scratch/cases/390/${name}/midway_zonal_mean.nc

#     rm -f ~/scratch/cases/800/${name}/??/midway_zonal_mean.nc
#     rm -f ~/scratch/cases/800/${name}/midway_zonal_mean.nc

#     rm -f ~/scratch/cases/1200/${name}/??/midway_zonal_mean.nc
#     rm -f ~/scratch/cases/1200/${name}/midway_zonal_mean.nc
# done