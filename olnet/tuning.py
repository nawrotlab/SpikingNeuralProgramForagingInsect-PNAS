import numpy as np
from math import pi as PI
from brian2 import *

def get_receptor_tuning(n_receptor_types, n_odors=1, receptors_per_odor=5, peak_rate=150):
    """
    compute a circular tuning profile over n_receptor_types for some number of odors (n_odors).
    Returns array of shape (n_receptor_types,n_odors)
    :param n_receptor_types: is equal to number of glomeruli
    :param n_odors: number of different odors
    :param receptors_per_odor: number of receptors activated by each odor
    :param peak_rate: peak rate of each 0.5 sin profile
    :return: ndarray
    """

    assert n_odors <= n_receptor_types, "no. of odors must be <= no. of receptor types"
    assert np.mod(receptors_per_odor, 2) != 2, "receptors per odor must be non-even number"

    y = np.zeros((n_receptor_types, n_odors), dtype=int)
    receptors = np.arange(0, n_receptor_types)

    for idx_odor, odor_id in enumerate(range(0, n_odors)):
        x = (np.mod((receptors - odor_id), n_receptor_types)) / (receptors_per_odor + 1.0)
        idx = np.logical_and(x > 0, x < 1)
        x[~idx] = 0
        r = peak_rate * np.sin(x * PI)  # tuning by 0.5 sin cycle with peak amplitude orn_peak_rate
        y[:, idx_odor] = r  # all activation profiles for different stimuli

    return y


def get_orn_tuning(receptor_tuning, odor_ids=None, n_orns=50, dim=1):
    """
    compute tuning of individual ORNs from receptor_type tuning. It's basically upsampling to number of ORN resolution
    Returns a matrix of shape (n_odors,n_receptor_types*n_orns,dim)
    :param receptor_tuning: matrix of receptor_type tuning
    :param odor_ids: odors to compute ORN tuning for. If None, it will be computed for ALL odors available in receptor_tuning
    :param n_orns: is equal to number of ORN neurons per receptor type/glomeruli
    :param dim: size of each ORN tuning - defaults to 1 (e.g. time dimension)
    :return: ndarray
    """

    if odor_ids is None:
        odor_ids = list(range(receptor_tuning.shape[1]))

    if (np.isscalar(odor_ids)):
        odor_ids = [odor_ids]

    n_receptors = receptor_tuning.shape[0] * n_orns
    Y = np.zeros((len(odor_ids), n_receptors, dim))
    for odor_idx, odor_id in enumerate(odor_ids):
        for rec_idx, rec_rate in enumerate(receptor_tuning[:, odor_idx]):
            Y[odor_idx, rec_idx * n_orns:((rec_idx + 1) * n_orns), :] = rec_rate

    return Y


def create_stimulation_matrix(trigger_ts, receptor_profiles, bg_rate=0, warmup_padding=None, odor_idx=None):
    """
    given a list of binary time-series, create a stimulation matrix using given receptor_profiles
    Returns array of shape (n_receptors, time-steps)
    :param trigger_ts: list of (multiple) binary time-series
    :param receptor_profiles: receptor profiles for each stimulation type (e.g. each binary time-series)
    :param bg_rate: background rate
    :param warmup_padding: optionally prepend some 'warmup_padding' timesteps
    :param odor_idx: only include triggers for specified odors
    :return: ndarray
    """
    assert (len(trigger_ts) == receptor_profiles.shape[0]), "number of binary stimulation protocols must be equal to receptor_profiles"
    n_receptors = receptor_profiles.shape[1]

    A = np.ones((n_receptors, len(trigger_ts[0]))) * bg_rate
    for i, odor in enumerate(trigger_ts):
        if odor_idx is not None and i in odor_idx:
            continue

        idx = np.where(odor == 1)
        for j in idx[0]:
            A[:, j] += receptor_profiles[i, :, 0]

    if warmup_padding is not None:
        warmup_pad = np.ones((n_receptors, warmup_padding)) * bg_rate
        return np.hstack((warmup_pad, A))

    return A


def gen_shot_noise(rate, T, tau=8, dt=0.001, dim=1, scale=1.0):
    """
    generate shot noise time-series by filtering Poisson noise.
    Returns ndarray of shape (dim, T * 1/dt)
    :param rate: Poisson rate
    :param T: duration
    :param tau: exp. filter time-constant
    :param dt: sampling / binning
    :param dim: number of time-series to generate
    :param scale: scale noise to max value of scale
    :return: TimedArray
    """
    X = np.zeros((dim, int(T / dt)))
    n_spikes = rate * T
    kernel_duration = 1
    t = np.arange(1e-8, kernel_duration, dt)

    assert X.shape[1] > t.shape[0], "duration must be larger than filter window"

    exp = lambda x: np.exp(-tau/x)
    kernel = exp(t)
    norm_kernel = kernel / np.sum(kernel)

    for d in range(dim):
        sp_times = np.random.uniform(0, T+2*kernel_duration, np.random.poisson(n_spikes))
        x, _ = np.histogram(sp_times, bins=np.arange(0, T + 2*kernel_duration + dt, dt))
        X[d, :] = np.convolve(x, norm_kernel, 'same')[len(t):len(t)+int(T / dt)]

    #X = np.linalg.norm(X, axis=1)[:, np.newaxis]
    return TimedArray((X/X.max(1)[:,np.newaxis]).T * scale, dt=dt * second)


def combine_noise_with_protocol(X_protocol: TimedArray, X_noise: TimedArray):
    """
    combine some (noise) signal with a stimulation protocol time-series.
    The signals is filtered by the protocol time-series
    :param X_protocol: stimulation protocol (binary/rect signal)
    :param X_noise:
    :return: TimedArray
    """
    dt = X_noise.dt
    T = X_protocol.values.shape[0] * X_protocol.dt
    t = np.arange(0, T, dt) * second
    print("T={} | dt={} | proto_dt: {} | X_noise: {} | X_protocol: {} | X_protocol(t): {}".format(T, dt, X_protocol.dt, X_noise.values.shape, X_protocol.values.shape, X_protocol(t).shape))
    mask = np.tile(X_protocol(t), (X_noise.values.shape[1], 1)).T
    L = (X_noise.values * mask)
    return TimedArray(L, dt=dt * second)


def gen_gauss_sequence(T, dt, std=1.5, mu=None, N=8):
    """
    generate sequence of discrete events that are approx. gaussian distributed
    :param T: duration of sequence in sec
    :param dt:
    :param std: std. dev. of gaussian
    :param mu: mean of gaussian (or default to T/2)
    :param N: no. of samples to draw
    :return: list
    """
    ts = np.zeros(int(T/dt))
    stim_pos = np.random.normal(T/2 if mu is None else mu, std, N + np.random.poisson(5)) # make no of samples a bit noisy
    stim_pos_idx = stim_pos/dt
    # add noise to duration of each stimulus
    stim_pos_noise = [[n, n+np.random.choice(2, 1,p=[.3,.7])[0], n-np.random.choice(2, 1, p=[.2,.8])[0]] for n in np.random.choice(stim_pos_idx.astype(np.int), len(stim_pos_idx))]
    flat_noise = [item for sublist in stim_pos_noise for item in sublist]
    stim_idx = np.union1d(stim_pos_idx.astype(np.int), flat_noise).astype(np.int)
    ts[stim_idx] = 1
    return ts
