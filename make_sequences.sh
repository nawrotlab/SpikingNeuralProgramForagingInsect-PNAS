#!/bin/bash
PYTHON_BIN="python3.6"
N_CPU=12

function defaultSparsity {
# 2% sparsity - default condition
$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name 'GaussianConnectivityHighSparsityAPL_15-0-3-15_10sec' \
-N 50 --odor_ids 0 --odor_ids 3 --odor_ids 15 --n_cpu $N_CPU -T 10 --stimulus_dt 5 --bg_noise_scale 0.0055 --pulse_rate 14 \
--gaussian 1 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
-o data/GaussianConnectivityHighSparsityAPL_15-0-3-15_10sec.mat

$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name PoisonPulseConnectivityHighSparsityAPL_0-3-15-10sec \
-N 50 --n_cpu $N_CPU --pulse_rate 8 -T 10 --stim_noise_scale 0.004 --bg_noise_scale 0.0055 --min_pulse_duration 0.001 --max_pulse_duration 0.2 \
--odor_ids 0 --odor_ids 3 --odor_ids 15 \
--gaussian 0 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
-o data/PoisonPulseConnectivityHighSparsityAPL_0-3-15-10sec.mat

$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name PoisonPulseConnectivityHighSparsityAPL_0-3-5-15-10sec \
-N 50 --n_cpu $N_CPU --pulse_rate 8 -T 10 --stim_noise_scale 0.004 --bg_noise_scale 0.0055 --min_pulse_duration 0.001 --max_pulse_duration 0.2 \
--odor_ids 0 --odor_ids 3 --odor_ids 5 --odor_ids 15 \
--gaussian 0 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
-o data/PoisonPulseConnectivityHighSparsityAPL_0-3-5-15-10sec.mat

$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name PoisonPulseConnectivityHighSparsityAPL_0-3-5-8-15-10sec \
-N 50 --n_cpu $N_CPU --pulse_rate 8 -T 10 --stim_noise_scale 0.004 --bg_noise_scale 0.0055 --min_pulse_duration 0.001 --max_pulse_duration 0.2 \
--odor_ids 0 --odor_ids 3 --odor_ids 5 --odor_ids 8 --odor_ids 15 \
--gaussian 0 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
-o data/PoisonPulseConnectivityHighSparsityAPL_0-3-5-8-15-10sec.mat

$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name PoisonPulseConnectivityHighSparsityAPL_0-3-8-15-10sec \
-N 50 --n_cpu $N_CPU --pulse_rate 8 -T 10 --stim_noise_scale 0.004 --bg_noise_scale 0.0055 --min_pulse_duration 0.001 --max_pulse_duration 0.2 \
--odor_ids 0 --odor_ids 3 --odor_ids 8 --odor_ids 15 \
--gaussian 0 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
-o data/PoisonPulseConnectivityHighSparsityAPL_0-3-8-15-10sec.mat
}

function connectivityMediumSparsity {
# 5% sparsity
$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name 'GaussianConnectivityMediumSparsityAPL_15-0-3-15_10sec' \
-N 50 --odor_ids 0 --odor_ids 3 --odor_ids 15 --n_cpu $N_CPU -T 10 --stimulus_dt 5 --bg_noise_scale 0.0055 --pulse_rate 14 \
--gaussian 1 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
--modelParams PNperKC=8.1 \
-o data/GaussianConnectivityMediumSparsityAPL_15-0-3-15_10sec.mat

$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name PoisonPulseConnectivityMediumSparsityAPL_0-3-15-10sec \
-N 50 --n_cpu $N_CPU --pulse_rate 8 -T 10 --stim_noise_scale 0.004 --bg_noise_scale 0.0055 --min_pulse_duration 0.001 --max_pulse_duration 0.2 \
--odor_ids 0 --odor_ids 3 --odor_ids 15 \
--gaussian 0 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
--modelParams PNperKC=8.1 \
-o data/PoisonPulseConnectivityMediumSparsityAPL_0-3-15-10sec.mat

$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name PoisonPulseConnectivityMediumSparsityAPL_0-3-5-15-10sec \
-N 50 --n_cpu $N_CPU --pulse_rate 8 -T 10 --stim_noise_scale 0.004 --bg_noise_scale 0.0055 --min_pulse_duration 0.001 --max_pulse_duration 0.2 \
--odor_ids 0 --odor_ids 3 --odor_ids 5 --odor_ids 15 \
--gaussian 0 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
--modelParams PNperKC=8.1 \
-o data/PoisonPulseConnectivityMediumSparsityAPL_0-3-5-15-10sec.mat

$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name PoisonPulseConnectivityMediumSparsityAPL_0-3-5-8-15-10sec \
-N 50 --n_cpu $N_CPU --pulse_rate 8 -T 10 --stim_noise_scale 0.004 --bg_noise_scale 0.0055 --min_pulse_duration 0.001 --max_pulse_duration 0.2 \
--odor_ids 0 --odor_ids 3 --odor_ids 5 --odor_ids 8 --odor_ids 15 \
--gaussian 0 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
--modelParams PNperKC=8.1 \
-o data/PoisonPulseConnectivityMediumSparsityAPL_0-3-5-8-15-10sec.mat

$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name PoisonPulseConnectivityMediumSparsityAPL_0-3-8-15-10sec \
-N 50 --n_cpu $N_CPU --pulse_rate 8 -T 10 --stim_noise_scale 0.004 --bg_noise_scale 0.0055 --min_pulse_duration 0.001 --max_pulse_duration 0.2 \
--odor_ids 0 --odor_ids 3 --odor_ids 8 --odor_ids 15 \
--gaussian 0 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
--modelParams PNperKC=8.1 \
-o data/PoisonPulseConnectivityMediumSparsityAPL_0-3-8-15-10sec.mat
}


function connectivityLowSparsity {
# ~10% sparsity
$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name 'GaussianConnectivityLowSparsityAPL_15-0-3-15_10sec' \
-N 50 --odor_ids 0 --odor_ids 3 --odor_ids 15 --n_cpu $N_CPU -T 10 --stimulus_dt 5 --bg_noise_scale 0.0055 --pulse_rate 14 \
--gaussian 1 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
--modelParams PNperKC=12 \
-o data/GaussianConnectivityLowSparsityAPL_15-0-3-15_10sec.mat

$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name PoisonPulseConnectivityLowSparsityAPL_0-3-15-10sec \
-N 50 --n_cpu $N_CPU --pulse_rate 8 -T 10 --stim_noise_scale 0.004 --bg_noise_scale 0.0055 --min_pulse_duration 0.001 --max_pulse_duration 0.2 \
--odor_ids 0 --odor_ids 3 --odor_ids 15 \
--gaussian 0 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
--modelParams PNperKC=12 \
-o data/PoisonPulseConnectivityLowSparsityAPL_0-3-15-10sec.mat

$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name PoisonPulseConnectivityLowSparsityAPL_0-3-5-15-10sec \
-N 50 --n_cpu $N_CPU --pulse_rate 8 -T 10 --stim_noise_scale 0.004 --bg_noise_scale 0.0055 --min_pulse_duration 0.001 --max_pulse_duration 0.2 \
--odor_ids 0 --odor_ids 3 --odor_ids 5 --odor_ids 15 \
--gaussian 0 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
--modelParams PNperKC=12 \
-o data/PoisonPulseConnectivityLowSparsityAPL_0-3-5-15-10sec.mat

$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name PoisonPulseConnectivityLowSparsityAPL_0-3-5-8-15-10sec \
-N 50 --n_cpu $N_CPU --pulse_rate 8 -T 10 --stim_noise_scale 0.004 --bg_noise_scale 0.0055 --min_pulse_duration 0.001 --max_pulse_duration 0.2 \
--odor_ids 0 --odor_ids 3 --odor_ids 5 --odor_ids 8 --odor_ids 15 \
--gaussian 0 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
--modelParams PNperKC=12 \
-o data/PoisonPulseConnectivityLowSparsityAPL_0-3-5-8-15-10sec.mat

$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name PoisonPulseConnectivityLowSparsityAPL_0-3-8-15-10sec \
-N 50 --n_cpu $N_CPU --pulse_rate 8 -T 10 --stim_noise_scale 0.004 --bg_noise_scale 0.0055 --min_pulse_duration 0.001 --max_pulse_duration 0.2 \
--odor_ids 0 --odor_ids 3 --odor_ids 8 --odor_ids 15 \
--gaussian 0 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
--modelParams PNperKC=12 \
-o data/PoisonPulseConnectivityLowSparsityAPL_0-3-8-15-10sec.mat
}


function weightMediumSparsity {
# 5% sparsity
$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name 'GaussianWeightMediumSparsityAPL_15-0-3-15_10sec' \
-N 50 --odor_ids 0 --odor_ids 3 --odor_ids 15 --n_cpu $N_CPU -T 10 --stimulus_dt 5 --bg_noise_scale 0.0055 --pulse_rate 14 \
--gaussian 1 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
--modelParams wPNKC=20 \
-o data/GaussianWeightMediumSparsityAPL_15-0-3-15_10sec.mat

$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name PoisonPulseWeightMediumSparsityAPL_0-3-15-10sec \
-N 50 --n_cpu $N_CPU --pulse_rate 8 -T 10 --stim_noise_scale 0.004 --bg_noise_scale 0.0055 --min_pulse_duration 0.001 --max_pulse_duration 0.2 \
--odor_ids 0 --odor_ids 3 --odor_ids 15 \
--gaussian 0 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
--modelParams wPNKC=20 \
-o data/PoisonPulseWeightMediumSparsityAPL_0-3-15-10sec.mat

$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name PoisonPulseWeightMediumSparsityAPL_0-3-5-15-10sec \
-N 50 --n_cpu $N_CPU --pulse_rate 8 -T 10 --stim_noise_scale 0.004 --bg_noise_scale 0.0055 --min_pulse_duration 0.001 --max_pulse_duration 0.2 \
--odor_ids 0 --odor_ids 3 --odor_ids 5 --odor_ids 15 \
--gaussian 0 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
--modelParams wPNKC=20 \
-o data/PoisonPulseWeightMediumSparsityAPL_0-3-5-15-10sec.mat

$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name PoisonPulseWeightMediumSparsityAPL_0-3-5-8-15-10sec \
-N 50 --n_cpu $N_CPU --pulse_rate 8 -T 10 --stim_noise_scale 0.004 --bg_noise_scale 0.0055 --min_pulse_duration 0.001 --max_pulse_duration 0.2 \
--odor_ids 0 --odor_ids 3 --odor_ids 5 --odor_ids 8 --odor_ids 15 \
--gaussian 0 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
--modelParams wPNKC=20 \
-o data/PoisonPulseWeightMediumSparsityAPL_0-3-5-8-15-10sec.mat

$PYTHON_BIN mkDataSet_DrosoArtificialStim.py --name PoisonPulseWeightMediumSparsityAPL_0-3-8-15-10sec \
-N 50 --n_cpu $N_CPU --pulse_rate 8 -T 10 --stim_noise_scale 0.004 --bg_noise_scale 0.0055 --min_pulse_duration 0.001 --max_pulse_duration 0.2 \
--odor_ids 0 --odor_ids 3 --odor_ids 8 --odor_ids 15 \
--gaussian 0 --gauss_mean 5 --gauss_std 1.5 --gauss_primary_odor_id 15 --gauss_rate_other 5 \
--modelParams wPNKC=20 \
-o data/PoisonPulseWeightMediumSparsityAPL_0-3-8-15-10sec.mat
}