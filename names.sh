#!/bin/bash

model_names=(
    mubofo-wind
    mubofo-wind-T
    mubofo-wind-T-noloc
    mubofo-wind-shear

    random-wind-T
    random-wind-shear

    WaveNet-wind
    WaveNet-wind-T
    WaveNet-wind-noloc
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

    if [[ $model_name != *${filter}* ]]; then
        continue
    fi

    s="${prefix}${model_name}${suffix}"
    if [ -n "$sep" ]; then
        out="${out}${s}${sep}"
    else
        echo $s
    fi

done

if [ -n "$sep" ]; then
    echo ${out%?}
fi