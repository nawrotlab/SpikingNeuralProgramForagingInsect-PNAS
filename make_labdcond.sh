#!/bin/bash
PYTHON_BIN="python3.6"
N_CPU=12
# default sparsity condition
$PYTHON_BIN mkDataSet_DrosoLabCondition.py --name LabCondConnectivityHighSparsityAPL_0-15-3sec \
-N 30 --n_cpu $N_CPU -T 3 --max_pulse_duration 0.5 --min_pulse_duration 0.1 \
--stim_noise_scale 0.004 --bg_noise_scale 0.0055 \
--odor_ids 0 --odor_ids 15 \
-o data/LabCondConnectivityHighSparsityAPL_0-15-3sec.mat

$PYTHON_BIN mkDataSet_DrosoLabCondition.py --name LabCondConnectivityMediumSparsityAPL_0-15-3sec \
-N 30 --n_cpu $N_CPU -T 3 --max_pulse_duration 0.5 --min_pulse_duration 0.1 \
--stim_noise_scale 0.004 --bg_noise_scale 0.0055 \
--odor_ids 0 --odor_ids 15 \
--modelParams PNperKC=8.1 \
-o data/LabCondConnectivityMediumSparsityAPL_0-15-3sec.mat

$PYTHON_BIN mkDataSet_DrosoLabCondition.py --name LabCondWeightMediumSparsityAPL_0-15-3sec \
-N 30 --n_cpu $N_CPU -T 3 --max_pulse_duration 0.5 --min_pulse_duration 0.1 \
--stim_noise_scale 0.004 --bg_noise_scale 0.0055 \
--odor_ids 0 --odor_ids 15 \
--modelParams wPNKC=20 \
-o data/LabCondWeightMediumSparsityAPL_0-15-3sec.mat

$PYTHON_BIN mkDataSet_DrosoLabCondition.py --name LabCondConnectivityLowSparsityAPL_0-15-3sec \
-N 30 --n_cpu $N_CPU -T 3 --max_pulse_duration 0.5 --min_pulse_duration 0.1 \
--stim_noise_scale 0.004 --bg_noise_scale 0.0055 \
--odor_ids 0 --odor_ids 15 \
--modelParams PNperKC=12 \
-o data/LabCondConnectivityLowSparsityAPL_0-15-3sec.mat
