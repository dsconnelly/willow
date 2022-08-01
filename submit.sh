#!/bin/bash

forests=(
    boosted-wind-T-e
    boosted-wind-Nsq-e
    boosted-wind-Nsq-pca-e
    boosted-wind-shear-e
    boosted-shear-Nsq-e
    random-wind-T-e
    random-wind-Nsq-e
)

for name in "${forests[@]}"; do
    sbatch -J ${name} willow.slurm train-forest data/control-1e7 models/${name}
done

networks=(
    WaveNet-wind-T-e
    WaveNet-wind-Nsq-e
)

for name in "${networks[@]}"; do
    sbatch -J ${name} willow.slurm train-network data/control-1e7 models/${name}
done