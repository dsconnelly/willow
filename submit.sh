#!/bin/bash

models=(
    boosted-wind-T
    boosted-wind-Nsq
    boosted-wind-Nsq-pca
    boosted-wind-shear
    boosted-shear-Nsq
    random-wind-T
    random-wind-Nsq
    WaveNet-wind-T
    WaveNet-wind-Nsq
)

suffix="g"
for model in "${models[@]}"; do
    name="${model}-${suffix}"
    sbatch -J ${name} willow.slurm train-emulator data/control-1e4 models/${name}
done