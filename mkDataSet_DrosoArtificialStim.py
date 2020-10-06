from olnet import run_sim, save_sim, save_sim_hdf5
from olnet.tuning import get_orn_tuning, get_receptor_tuning, create_stimulation_matrix, gen_shot_noise, combine_noise_with_protocol, gen_gauss_sequence
from brian2 import *
import numpy as np
import olnet.models.droso_mushroombody_apl as droso_mb
from olnet import AttrDict
import sys,argparse,time, os
import traceback

def current_milli_time():
    return int(round(time.time() * 1000))

def flatten(a,b):
    return a+b

def divide(lst, mean_rate, split_size):
    it = iter(lst)
    from itertools import islice
    size = len(lst)
    for i in range(split_size - 1,0,-1):
        s = max(2, int(np.random.poisson(mean_rate, size=1)))
        yield list(islice(it,0,s))
        size -= s
    yield list(it)


def gen_pulsed_gauss_stimulus(odor_ids, T, dt, mu, std, n_pulses, primary_odor_id, n_stim=1, allow_overlap=False):
    """
    generate a pulsed stimulus that simulates a gaussian plume cone profile
    Returns tuple of (stimulus,n_pulses,pulse_times)
    :param odor_ids:
    :param T:
    :param dt:
    :param mu: mean of gaussian profile (or T/2 if None)
    :param std: std. dev. of gaussian
    :param n_pulses: tuple(n_primary,n_bg) number of pulses to generate for primary_odor_id and all other background odors
    :param primary_odor_id: the primary odor_id that should use the gaussian profile. all other odor_ids will be background distractors
    :param n_stim:
    :param allow_overlap:
    :return:
    """

    if type(n_pulses) == tuple:
        n_primary,n_other = n_pulses
    else:
        n_primary = n_pulses
        n_other = n_pulses // 3

    X = np.zeros((n_stim, int(T / dt)))
    X[primary_odor_id, :] = gen_gauss_sequence(T, dt, std, mu, n_primary)

    for odor_id in odor_ids:
        if odor_id != primary_odor_id:
            bg_idx = np.random.choice(X.shape[1], n_other)
            X[odor_id, bg_idx] = 1
            if allow_overlap is False:
                overlap_idx = np.where(X[odor_id, :] == X[primary_odor_id, :])[0]
                X[odor_id, overlap_idx] = 0

    X_prime = np.c_[np.zeros((n_stim, 1)), X]
    pulse_times = [(np.where(np.diff(X_prime[n, :]) == 1)[0] * dt).tolist() for n in range(n_stim)]
    return X, [len(p) for p in pulse_times], pulse_times

def gen_pulsed_stimulus(odor_ids, T, dt, pulse_rate, pulse_duration=(0.1, 0.5), n_stim=1, allow_overlap=False):
    """
    generate a pulsed stimulus where no. of stimuli are poisson distr. with pulse_rate
    and gaps between subsequent pulses are drawn uniformly. Returns tuple of (stimulus,n_pulses,pulse_times)
    :param: odor_ids:
    :param T:
    :param dt:
    :param pulse_rate:
    :param pulse_duration:
    :param n_stim:
    :param allow_overlap:
    :return: tuple
    """
    from functools import reduce
    n_bins = int(T / dt) * n_stim
    pulses = np.random.uniform(pulse_duration[0], pulse_duration[1], size=np.random.poisson(pulse_rate))
    pulse_bins = (pulses / dt).astype(np.int)
    X = np.zeros((n_stim, int(T / dt)))

    spaced_pulses = [[1] * p for p in pulse_bins.tolist()]
    print("n_pulses: {} | spaced_pulses: {}".format(len(pulse_bins), len(spaced_pulses)))
    pad_x = [0] * int((n_bins // n_stim) - pulse_bins.sum())
    pad_sizes = np.random.randint(10, int((len(pad_x) - len(pulse_bins)) // len(spaced_pulses)),
                                  size=len(spaced_pulses)).tolist()
    ptr = 0
    start = 0
    pulse_times = [[] for _ in range(n_stim)]
    for k, s in enumerate(spaced_pulses):
        stim_idx = np.random.choice(odor_ids, size=1, p=[1. / len(odor_ids) for _ in odor_ids])[0]
        print("assigning pulse {}/{} to odor_id {}".format(k, len(spaced_pulses), stim_idx))
        stop = pad_sizes.pop(0)
        pad = pad_x[start:start + stop]
        #print("pulse #{} len: {} n_pad: {}".format(k, len(s), len(pad)))
        ptr += len(pad)
        X[stim_idx, ptr:ptr + len(s)] = s
        #print("stim_idx: {} / {} / {}".format(stim_idx, n_stim, pulse_times))
        pulse_times[stim_idx].append(ptr * dt)
        ptr += len(s)
        start = stop

    X_prime = np.c_[np.zeros((n_stim, 1)), X]
    return X, [len(np.where(np.diff(X_prime[n, :]) == 1)[0]) for n in range(n_stim)], pulse_times

def gen_shared_params(model_params, params, neuron_models):
    # autom. create shared model params for each neuron model type
    for v in params:
        for n in neuron_models:
            k = '{}{}'.format(v, n)
            if k not in model_params:
                model_params.update({k: model_params[v]})
    return model_params


def gen_shotnoise_input(dt, warmup_time, pulse_stim, n_receptors, N_glo, odor_ids, ORNperGlo, receptors_per_odor, stimulus_rate, bg_rate, stim_scale=0.003, bg_scale=0.001):
    #print("pulse_stim: {}".format(pulse_stim.shape))
    simtime = ((warmup_time/second) + (dt/second) * pulse_stim.shape[1]) * second
    print("ORNs={} glumeruli={} ORNperGlu={} receptors_per_odor={}, n_receptors={}, odor_ids={}".format(n_receptors, N_glo, ORNperGlo,
                                                                                         receptors_per_odor,
                                                                                         n_receptors, odor_ids))
    pad_n = int(warmup_time / dt)
    # TODO: support more than 2 odors
    y = []
    for odor_idx in odor_ids:
        y.append(np.array([0]*pad_n + pulse_stim[odor_idx, :].tolist())) # odor A pulse stim
    #y2 = np.array([0]*pad_n + pulse_stim[1, :].tolist()) # odor B pulse stim

    ORN_noise = gen_shot_noise(stimulus_rate, simtime / second, tau=0.6, dt=dt/second, dim=n_receptors, scale=stim_scale)

    S = get_receptor_tuning(N_glo, N_glo, receptors_per_odor, peak_rate=stimulus_rate) / stimulus_rate
    M = get_orn_tuning(S, n_orns=ORNperGlo)
    M_prime = M[odor_ids, :, :]

    A = (gen_shot_noise(bg_rate, simtime / second, tau=0.5, dt=dt / second, dim=n_receptors, scale=bg_scale).values)

    # subsequently add/superimpose stimuli of all odors
    #print("y={}".format(len(y)))
    for i,odor_idx in enumerate(odor_ids):
        y1 = combine_noise_with_protocol(TimedArray(y[i], dt=dt), ORN_noise)
        A += (y1.values * np.tile(M_prime[i], (1, y1.values.shape[0])).T)


    #A = (gen_shot_noise(bg_rate, simtime / second, tau=0.9, dt=dt/second, dim=n_receptors, scale=bg_scale).values) \
    #    + ((y1.values * np.tile(M_prime[0], (1, y1.values.shape[0])).T) + (y2.values * np.tile(M_prime[1], (1, y2.values.shape[0])).T))

    print("created stimulation matrix for {} odors: {}".format(len(odor_ids), A.shape))
    stimulus = TimedArray(A * uA, dt=dt)
    print("created stimulus TimedArray: {} warmup: {}".format(stimulus.values.shape, warmup_time))
    return simtime, stimulus, M_prime


def run_model(model_params, N_glo, ORNperGlo, N_KC, simtime, stimulus, dt = 0.1 * ms, network_seed=42):
    # use fixed random seed to build same network arch.
    np.random.seed(network_seed)
    seed(network_seed)

    model_params = gen_shared_params(model_params, ['C', 'gL', 'EL', 'Vt', 'Vr', 'tau_Ia'], ['ORN', 'PN', 'LN', 'KC', 'APL'])
    model_params.update({'stimulus': stimulus})


    NG, c = droso_mb.network(model_params,
                             None,
                             droso_mb.model_ORN,
                             droso_mb.model_PN,
                             droso_mb.model_LN,
                             droso_mb.model_KC,
                             droso_mb.model_APL,
                             wORNinputORN=1 * model_params['w0'],
                             wORNPN=1.1282 * model_params['w0'],
                             wORNLN=1 * model_params['w0'],
                             wLNPN=2.5 * model_params['w0'],  # enable lateral inhib.
                             wPNKC=double(model_params['wPNKC']) * model_params['w0'],
                             wKCAPL=double(model_params['wKCAPL']) * model_params['w0'],
                             wAPLKC=double(model_params['wAPLKC']) * model_params['w0'],
                             N_glu=N_glo,
                             ORNperGlu=ORNperGlo,
                             N_KC=N_KC,
                             PNperKC=double(model_params['PNperKC']),
                             V0min=model_params['EL'],
                             V0max=model_params['Vt'],
                             apl_delay=model_params['apl_delay'])

    var_mons = [
        ('ORN', ('v', 'g_i', 'g_e'), [360]),
        ('PN', ('v', 'g_i', 'g_e'), [15]),
        ('LN', ('v', 'g_i', 'g_e'), [15]),
        ('APL', ('v', 'g_i', 'g_e'), [0])
    ]

    return run_sim(model_params, NG, c, simtime, sim_dt=dt,
                                                        spike_monitors=['ORN', 'PN', 'LN', 'KC', 'APL'],
                                                        rate_monitors=['ORN', 'PN', 'LN', 'KC', 'APL'],
                                                        state_monitors=var_mons)




def worker(args):
    (id, name, seed, model_params, args, plot) = args
    np.random.seed(seed)

    t_start = current_milli_time()
    N_glo = 52
    ORNperGlo = (2080 // N_glo)  # Droso: roughly 2000 ORNs total
    N_KC = 2000                 # droso: 2000
    n_receptors = N_glo * ORNperGlo  # * model_params['orn_input_multiplier']
    receptors_per_odor = 15
    warmup_time = args.warmup_time * second # 2 * second
    sim_dt = 0.1 * ms
    stim_dt = args.stimulus_dt * ms # 1*ms # time-resolution for stimulus TimedArray
    bg_rate = args.bg_rate
    stimulus_rate = args.stimulus_rate
    T = args.T
    stim_noise_scale = args.stim_noise_scale # 0.003
    bg_noise_scale = args.bg_noise_scale #0.001

    model_params.update({
        'seed': seed,
        'T': T,
        'pulse_rate': args.pulse_rate,
        'min_pulse_duration': args.min_pulse_duration,
        'max_pulse_duration': args.max_pulse_duration,
        'stim_noise_scale': stim_noise_scale,
        'bg_noise_scale': bg_noise_scale,
        'stim_dt': stim_dt / second,
        'noise_bg_rate': bg_rate,
        'noise_stim_rate': stimulus_rate,
        'N_KC': N_KC,
        'ORNperGlo': ORNperGlo,
        'n_receptors': n_receptors,
        'receptors_per_odor': receptors_per_odor
    })

    print("worker[{}] started odor_ids: {} ...".format(id, args.odor_ids))
    pulse_stim, rewards = None, None

    # loop - to catch rare cases where stimulus could not be generated
    while (pulse_stim is None):
        try:
            if args.gaussian <= 0:
                pulse_stim, rewards, pulse_times = gen_pulsed_stimulus(args.odor_ids, T, stim_dt / second, args.pulse_rate,
                                                      pulse_duration=(args.min_pulse_duration, args.max_pulse_duration),
                                                      n_stim=N_glo)
            else:
                pulse_stim, rewards, pulse_times = gen_pulsed_gauss_stimulus(args.odor_ids, T, stim_dt / second,
                                                                             args.gauss_mean, args.gauss_std,
                                                                             (args.pulse_rate,args.gauss_rate_other),
                                                                             int(args.gauss_primary_odor_id),
                                                                       n_stim=N_glo)
        except Exception as e:
            traceback.print_exc()
            pulse_stim, rewards, pulse_times = None, None, []

    simtime, stimulus, M = gen_shotnoise_input(stim_dt, warmup_time, pulse_stim, n_receptors, N_glo, args.odor_ids, ORNperGlo, receptors_per_odor,
                            stimulus_rate, bg_rate, stim_scale=stim_noise_scale, bg_scale=bg_noise_scale)


    spikemons, pop_mons, state_mons, var_mons = run_model(model_params, N_glo, ORNperGlo, N_KC, (T + args.warmup_time) * second, stimulus, sim_dt, args.network_seed)
    t_stop = current_milli_time()
    print("worker[{}] finished (took {} sec)".format(id, (t_stop-t_start)/1000))

    model_params.update({'rewards': rewards})
    model_params.update({'stimulation_times': pulse_times})
    model_params.pop('stimulus', None)  # TimedArray is not pickle-able - remove it

    if plot:
        fileName = "sim-{}-{}".format(id, seed)
        data = save_sim("cache/{}/{}.npz".format(name, fileName),
                        model_params,
                        spikemons, pop_mons, state_mons, simtime, warmup_time, sim_dt,
                        stimulus=np.flipud(stimulus.values.T),
                        tuning=M,
                        stimulus_times=pulse_times,
                        n_receptors=n_receptors
                        )

    if plot:
        from olnet.plotting.figures import figure1
        f = figure1(data)
        f.savefig("figures/{}/{}.png".format(name, fileName), dpi=f.dpi)
        print("worker[{}] saved figure: figures/{}/{}.png".format(id, name, fileName))


    # align spiketrains to warmup offset
    sp_trains_aligned = {}
    for k, v in spikemons.items():
        trial_sp = []
        for s in v.spike_trains().values():
            sp_times = (s / second) - args.warmup_time
            trial_sp.append(list(sp_times))
        sp_trains_aligned[k] = trial_sp

    spikeData = AttrDict({
        k: AttrDict({'count': v.count[:],
                     't': (v.t[:] / second),
                     't_aligned': (v.t[:] / second) - args.warmup_time,
                     'i': v.i[:],
                     'spike_trains': v.spike_trains(),
                     'spike_trains_aligned': sp_trains_aligned[k]}) for k, v in spikemons.items()
    })

    return (id, rewards, spikeData, pulse_times, (t_stop-t_start))



if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    import scipy.io as scpio

    argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Generate data set of KC spike-times using drosoMB model and artificially generated stimulus sequences')

    parser.add_argument('-n', '--name', type=str, nargs='?', help='name of data-set')
    parser.add_argument('-N', '--N', type=int, nargs='?', help = 'number of samples to generate', default=10)
    parser.add_argument('--network_seed', type=int, nargs='?', help='RNG seed used to build network model', default=42)
    parser.add_argument('--odor_ids', type=int, action='append', help='indices of odors to use', required=True)
    parser.add_argument('--n_cpu', type=int, nargs='?', help = 'no of CPUs to use for parallel simulations', default=4)
    parser.add_argument('--bg_rate', type=int, nargs='?', help = 'background shot noise poisson rate', default=300)
    parser.add_argument('--stimulus_rate', type=int, nargs='?', help = 'stimulus shot noise poisson rate', default=300)
    parser.add_argument('-T', type=float, nargs='?', help = 'stimulus duration (in seconds)', default=10)
    parser.add_argument('--warmup_time', type=float, nargs='?', help = 'duration of warmup phase (in seconds)', default=2)
    parser.add_argument('--stimulus_dt', type=float, nargs='?', help = 'dt of stimulus TimedArray (in ms)', default=0.5)
    parser.add_argument('--pulse_rate', type=int, nargs='?', help = 'max. number of pulses within a sequence', default=8)
    parser.add_argument('--max_pulse_duration', type=float, nargs='?', help = 'max. duration of a single pulse (in seconds)', default=0.5)
    parser.add_argument('--min_pulse_duration', type=float, nargs='?', help = 'max. duration of a single pulse (in seconds)', default=0.1)
    parser.add_argument('--stim_noise_scale', type=float, nargs='?', help = 'scale of shot-noise for stimulus', default=0.004)
    parser.add_argument('--bg_noise_scale', type=float, nargs='?', help = 'scale of shot-noise for background activity', default=0.0055)

    parser.add_argument('--gaussian', type=int, nargs='?', help='whether to use gaussian profile', default=0)
    parser.add_argument('--gauss_mean', type=float, nargs='?', help='mean of gaussian profile',required=True)
    parser.add_argument('--gauss_std', type=float, nargs='?', help='mean of gaussian profile', default=1.5)
    parser.add_argument('--gauss_primary_odor_id', type=int, nargs='?', help='primary odor_id to use for gaussian profile', required=True)
    parser.add_argument('--gauss_rate_other', type=int, nargs='?', help='number of stimuli to draw for other odor_ids (uniform)',default=3)

    parser.add_argument('-o', '--outfile', nargs='?', type=str, help = 'output filaneme for MAT file')
    parser.add_argument("--modelParams", action='append', type=lambda kv: kv.split("="), dest='customModelParams')

    args = parser.parse_args()


    os.makedirs("cache/{}".format(args.name), exist_ok=True)
    os.makedirs("figures/{}".format(args.name), exist_ok=True)

    print(args)

    model_params = {
        # 'orn_input_multiplier': 1,  # distribute total poisson rate over 10 indep. processes
        # Neuron Parameters
        'C': 289.5 * pF,
        'gL': 28.95 * nS,
        'EL': -70 * mV,
        'Vt': -57 * mV,
        'Vr': -70 * mV,
        'tau_ref': 5 * ms,
        # APL parameters
        'VtAPL': -50 * mV,
        'VrAPL': -55 * mV,
        'ELAPL': -55 * mV,
        'gLAPL': 0.5 * nS,
        'CAPL': 10 * pF,
        'apl_delay': 0.2 * ms,
        # Synaptic Parameters
        'Ee': 0 * mV,
        'Ei': -75 * mV,
        'EIa': -90 * mV,  # reversal potential
        'tau_syn_e': 2 * ms,
        'tau_syn_i': 10 * ms,
        'tau_Ia': 1000 * ms,  # adaptation conduct. time constatnt
        'tau_IaKC': 50 * ms,  # adaptation time constant for KCs
        # Weights
        'w0': 1 * nS,
        # Adaptation Parameters
        'bORN': 2 * nS,
        'bKC': 5 * nS,
        'bLN': 0 * nS,
        'bPN': 0 * nS,
        'D': 0.005,
        'PNperKC': 6, # this will achieve ~8% KC activity
        'wPNKC': 14,
        'wKCAPL': 3,
        'wAPLKC': 3
    }

    if args.customModelParams is not None:
        model_params.update(args.customModelParams)
    else:
        args.customModelParams = {}
        
    print(model_params)

    trial_ids = []
    samples = []
    samples_alt = []
    odor_ids = []
    rewards = []
    stim_times = []
    durations = []
    warmup = args.warmup_time

    worker_args = [(id, args.name, seed, model_params, args, id in list(range(5))) for id,seed in enumerate(np.random.randint(142, size=args.N))]
    with ProcessPoolExecutor(max_workers=args.n_cpu) as executor:
        #result = executor.map(worker, worker_args)
        for params, result in zip(worker_args, executor.map(worker, worker_args)):
            task_id,reward,sp_data,pulse_times,duration = result
            rewards.append(reward)
            trial_ids.append(task_id)
            trial_sp = []
            for sp in sp_data.KC.spike_trains_aligned:
                sp_times = filter(lambda s: s >= 0.0, sp)  # only spikes AFTER warmup
                trial_sp.append(list(sp_times))

            samples.append(trial_sp)
            odor_ids.append(args.odor_ids)
            samples_alt.append(dict({'t': sp_data.KC.t_aligned, 'i': sp_data.KC.i}))
            durations.append(duration)
            stim_times.append(pulse_times)
            print("{} finished - avg. duration: {}".format(task_id, np.array(durations).mean()))


    output = {
        'trial_ids': trial_ids,
        'targets': rewards,
        'odor_ids': odor_ids,
        'stimulus_times': stim_times,
        'trials': samples,
        'trials_tuples': samples_alt,
        'T_trial': args.T,
        'N_trials': len(rewards)
    }

    scpio.savemat(args.outfile, {'data': output, 'args': args})
    print("saved to MATLAB file: {}".format(args.outfile))
    npzFile = args.outfile[:-4] + ".npz"
    np.savez(npzFile, data=output, args=args)
    print("saved to NPZ file: {}".format(npzFile))