#!/bin/bash

sbatch -J boosted-wind-T-c forests.slurm
sbatch -J boosted-wind-Nsq-c forests.slurm
sbatch -J boosted-wind-Nsq-pca-c forests.slurm
sbatch -J boosted-wind-shear-c forests.slurm
sbatch -J boosted-shear-Nsq-c forests.slurm

sbatch -J random-wind-T-c forests.slurm
sbatch -J random-wind-Nsq-c forests.slurm

sbatch -J WaveNet-wind-T-c networks.slurm
sbatch -J WaveNet-wind-Nsq-c networks.slurm