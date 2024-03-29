#!/bin/bash

model_names=(
    mubofo-wind
    mubofo-wind-T
    mubofo-wind-T-noloc
    mubofo-wind-shear
    mubofo-shear-T

    # mubofo-wind-T-repeat1
    # mubofo-wind-T-repeat2
    # mubofo-wind-T-repeat3

    # mubofo-wind-T-lat_scale_0.01
    # mubofo-wind-T-lat_scale_0.85
    mubofo-wind-T-lat_scale_1.55
    # mubofo-wind-T-lat_scale_2.05

    random-wind-T
    random-wind-shear

    WaveNet-wind
    WaveNet-wind-T
    WaveNet-wind-noloc

    # WaveNet-wind-T-repeat1
    # WaveNet-wind-T-repeat2
    # WaveNet-wind-T-repeat3
)

prefix=$1
suffix=$2
filter=$3
sep=$4

if [ -n "$prefix" ]; then
    prefix="${prefix}/"
fi

if [ -n "$suffix" ]; then
    suffix="-${suffix}"
fi

if [ -n "$sep" ]; then
    out=""
    if [[ $prefix == *"cases"* ]]; then
        out="${prefix}ad99${sep}"
    fi
fi

for model_name in ${model_names[@]}; do
    s="${prefix}${model_name}${suffix}"
    if [[ $s != *${filter}* ]]; then
        continue
    fi

    if [ -n "$sep" ]; then
        out="${out}${s}${sep}"
    else
        echo $s
    fi

done

if [ -n "$sep" ]; then
    echo ${out%?}
fi