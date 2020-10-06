from olnet import run_sim, save_sim, save_sim_hdf5
from olnet.tuning import get_orn_tuning, get_receptor_tuning, create_stimulation_matrix, gen_shot_noise, combine_noise_with_protocol
from brian2 import *
import numpy as np
import olnet.models.droso_mushroombody_apl as droso_mb
from olnet import AttrDict
import sys,argparse,time, os
import random
import ast

def current_milli_time():
    return int(round(time.time() * 1000))


def gen_pulsed_stimulus(T, dt, odor_idx, pulse_duration=(0.1, 0.5), n_stim=1):
    """
    generate a single pulsed stimulus that is randomly positioned within [0,T].
    Pulse duration is randomly sampled from given bounds
    Returns tuple of (stimulus,n_pulses,pulse_times)
    :param T:
    :param dt:
    :param odor_idx: index of specific odor to generate stimulus for
    :param pulse_duration:
    :param n_stim: number of total stimulation types (e.g. odors)
    :return: tuple
    """
    # randomly sample pulse duration
    pulses = np.random.uniform(pulse_duration[0], pulse_duration[1], size=100)
    pulse_bins = (pulses / dt).astype(np.int).tolist()
    random.shuffle(pulse_bins)
    X = np.zeros((n_stim, int(T / dt)))

    # randomly position pulse - use poisson to have more variability in positioning
    n_bins = int(T / dt)
    spaced_pulses = [[1] * pulse_bins[0]]
    start_bins = np.random.randint(0, n_bins-int(0.1/dt), size=200).astype(np.int).tolist()
    #start_bins = (np.random.poisson(int(T * 100), size=10) / 100 / dt).astype(np.int)
    #print("start_bins: {}".format(start_bins))
    start_bins = list(filter(lambda p: (p+pulse_bins[0]) < (n_bins-5), start_bins))
    random.shuffle(start_bins)
    #print("start_bins: {} | {} sec".format(start_bins, np.array(start_bins)*dt))
    start_bin = start_bins[0]
    print("pulse offset: {}sec | duration: {}sec".format(start_bin * dt, pulse_bins[0] * dt))

    pulse_times = [[] for _ in range(n_stim)]
    for k, s in enumerate(spaced_pulses):
        X[odor_idx, start_bin:start_bin + len(s)] = s
        pulse_times[odor_idx].append(start_bin * dt)

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


def gen_shotnoise_input(protocol_dt, dt, warmup_time, pulse_stim, n_receptors, N_glo, odor_ids, ORNperGlo, receptors_per_odor, stimulus_rate, bg_rate, stim_scale=0.003, bg_scale=0.001):
    #print("pulse_stim: {}".format(pulse_stim.shape))
    simtime = ((warmup_time/second) + (protocol_dt/second) * pulse_stim.shape[1]) * second
    print("ORNs={} glumeruli={} ORNperGlu={} receptors_per_odor={}, n_receptors={}, odor_ids={}, simtime={}".format(n_receptors, N_glo, ORNperGlo,
                                                                                         receptors_per_odor,
                                                                                         n_receptors, odor_ids,simtime))
    pad_n = int(warmup_time / protocol_dt)
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
        y1 = combine_noise_with_protocol(TimedArray(y[i], dt=protocol_dt), ORN_noise)
        A += (y1.values * np.tile(M_prime[i], (1, y1.values.shape[0])).T)


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
    (id, stimulus_id, network_id, name, seed, odor_ids, stimulus_protocol, model_params, args, plot) = args

    N_odors = 1
    a = np.array(stimulus_protocol)
    stim_protocols = [stimulus_protocol]
    T = len(stimulus_protocol) * args.dt

    if len(a.shape) > 1:
        N_odors = a.shape[0]
        T = a.shape[1] * args.dt
        stim_protocols = stimulus_protocol
    else:
        a = np.array(stim_protocols)

    assert N_odors == len(odor_ids), "number of stimulation protocols ({}) must be equal to number of odors ({})".format(N_odors, len(odor_ids))

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
    stim_noise_scale = args.stim_noise_scale # 0.003
    bg_noise_scale = args.bg_noise_scale #0.001

    model_params.update({
        'seed': seed,
        'T': T,
        'stimulus_protocol': stimulus_protocol,
        'stimulus_protocol_dt': args.dt,
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

    print("worker[{} stimulus_id {} (n_odors: {}) network_id: {}] started ...".format(id, stimulus_id, N_odors, network_id))

    X = np.zeros((N_glo, a.shape[1]))
    for j,odor_idx in enumerate(odor_ids):
        X[odor_idx, :] = stim_protocols[j]

    X_prime = np.c_[np.zeros((N_glo, 1)), X]
    rewards = [len(np.where(np.diff(X_prime[n, :]) == 1)[0]) for n in range(N_glo)]
    pulse_times = [(np.where(X[n,:] == 1)[0] * args.dt).tolist() for n in range(N_glo)]

    simtime, stimulus, M = gen_shotnoise_input(args.dt * second, stim_dt, warmup_time, X, n_receptors, N_glo, odor_ids,
                                               ORNperGlo, receptors_per_odor,
                                               stimulus_rate, bg_rate, stim_scale=stim_noise_scale,
                                               bg_scale=bg_noise_scale)


    spikemons, pop_mons, state_mons, var_mons = run_model(model_params, N_glo, ORNperGlo, N_KC, (T + args.warmup_time) * second, stimulus, sim_dt, network_id)
    t_stop = current_milli_time()
    print("worker[{}] finished (took {} sec)".format(id, (t_stop-t_start)/1000))

    model_params.update({'rewards': rewards})
    model_params.update({'stimulation_times': pulse_times})
    model_params.pop('stimulus', None)  # TimedArray is not pickle-able - remove it

    if plot:
        fileName = "sim-{}-stim_id-{}-net_id-{}-seed-{}".format(id,stimulus_id, network_id, seed)
        data = save_sim("cache/{}/{}.npz".format(name, fileName),
                        model_params,
                        spikemons, pop_mons, state_mons, simtime, warmup_time, sim_dt,
                        stimulus=np.flipud(stimulus.values.T),
                        tuning=M,
                        stimulus_times=pulse_times,
                        n_receptors=n_receptors,
                        stimulus_protocol=stimulus_protocol
                        )

    if plot:
        from olnet.plotting.figures import figure1
        f = figure1(data)
        f.savefig("figures/{}/{}.png".format(name, fileName), dpi=f.dpi)
        print("worker[{}] saved figure: figures/{}/{}.png".format(id, name, fileName))

    # align spiketrains to warmup offset
    sp_trains_aligned = {}
    for k,v in spikemons.items():
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

    return (id, stimulus_id, network_id, seed, odor_ids, rewards, spikeData, pulse_times, T, (t_stop-t_start))


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    import scipy.io as scpio

    argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Generate data set of KC spike-times using drosoMB model and custom stimulation protocol')

    parser.add_argument('-n', '--name', type=str, nargs='?', help='name of data-set')
    parser.add_argument('-N', '--N', type=int, nargs='?', help = 'number of samples to generate for each stimulation', default=5)
    parser.add_argument('--network_seeds', type=list, nargs='?', help='RNG seed(s) used to build network model. If list > 0 multiple indep. networks will be simulated', default=[42])
    parser.add_argument('--protocols', type=arg_as_list, action='append', help='stimulation protocols to run.(single odor: [1,0,0], 2 odors: [[1,1,0,0],[0,1,1,0]])', required=True)
    parser.add_argument('--odor_ids', type=arg_as_list, action='append', help='odor_ids used in each stimulus (single odor: [0] 2 odors: [5, 15]])', required=True)
    parser.add_argument('--n_cpu', type=int, nargs='?', help = 'no of CPUs to use for parallel simulations', default=1)
    parser.add_argument('--bg_rate', type=int, nargs='?', help = 'background shot noise poisson rate', default=300)
    parser.add_argument('--stimulus_rate', type=int, nargs='?', help = 'stimulus shot noise poisson rate', default=300)
    parser.add_argument('-dt', type=float, nargs='?', help = 'dt of stimulation protocol (in seconds)', default=0.5)
    parser.add_argument('--warmup_time', type=float, nargs='?', help = 'duration of warmup phase (in seconds)', default=2)
    parser.add_argument('--stimulus_dt', type=float, nargs='?', help = 'dt of stimulus TimedArray (in ms)', default=0.5)
    parser.add_argument('--stim_noise_scale', type=float, nargs='?', help = 'scale of shot-noise for stimulus', default=0.004)
    parser.add_argument('--bg_noise_scale', type=float, nargs='?', help = 'scale of shot-noise for background activity', default=0.0055) # use 0.0056 for more noise
    parser.add_argument('-o', '--outfile', nargs='?', type=str, help = 'output filename for MAT file')
    parser.add_argument("--modelParams", action='append', type=lambda kv: kv.split("="), dest='customModelParams')

    args = parser.parse_args()

    os.makedirs("cache/{}".format(args.name), exist_ok=True)
    os.makedirs("figures/{}".format(args.name), exist_ok=True)

    if os.path.isfile(args.outfile):
        print("skipped - cache file exists at: {}".format(args.outfile))
        exit(0)

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

    samples = []
    trial_ids = []
    samples_alt = []
    rewards = []
    stimulus_ids = []
    odor_ids = []
    network_ids = []
    stim_times = []
    trial_durations = []
    durations = []
    warmup = args.warmup_time

    worker_args = []
    for stim_id,(stim_protocol,protocol_odor_ids) in enumerate(zip(args.protocols,args.odor_ids)):
        for network_id in args.network_seeds:
            worker_args.extend([(id, stim_id, network_id, args.name, seed, protocol_odor_ids, stim_protocol, model_params, args, id in list(range(5))) for id,seed in enumerate(np.random.randint(142, size=args.N))])

    with ProcessPoolExecutor(max_workers=args.n_cpu) as executor:
        #result = executor.map(worker, worker_args)
        for params, result in zip(worker_args, executor.map(worker, worker_args)):
            task_id,stim_id,net_id,_,protocol_odor_ids,reward,sp_data,pulse_times,T,duration = result
            rewards.append(reward)
            network_ids.append(net_id)
            odor_ids.append(protocol_odor_ids)
            stimulus_ids.append(stim_id)
            trial_ids.append(task_id)
            trial_durations.append(T)
            trial_sp = []
            for sp in sp_data.KC.spike_trains_aligned:
                sp_times = filter(lambda s: s >= 0.0, sp)  # only spikes AFTER warmup
                trial_sp.append(list(sp_times))

            samples.append(trial_sp)
            samples_alt.append(dict({'t': sp_data.KC.t_aligned, 'i': sp_data.KC.i}))
            durations.append(duration)
            stim_times.append(pulse_times)
            print("{} finished - avg. duration: {}".format(task_id, np.array(durations).mean()))


    output = {
        'trial_ids': trial_ids,
        'targets': rewards,
        'odor_ids': odor_ids,
        'stimulus_ids': stimulus_ids,
        'network_ids': network_ids,
        'stimulus_times': stim_times,
        'trials': samples,
        'trials_tuples': samples_alt,
        'T_trials': trial_durations
    }

    scpio.savemat(args.outfile, {'data':output, 'args': args})
    print("saved to MATLAB file: {}".format(args.outfile))
    npzFile = args.outfile[:-4] + ".npz"
    np.savez(npzFile, data=output, args=args)
    print("saved to NPZ file: {}".format(npzFile))