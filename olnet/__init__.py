from brian2 import *
import inspect
#from collections import namedtuple
#__all__ = ["echo", "surround", "reverse"]

class AttrDict(dict):
    """
    dict subclass which allows access to keys as attributes: mydict.myattr
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_args_from_dict(fn, params):
    """
    extract function parameters by name from a dict
    :param fn:
    :param params:
    :return: dict
    """
    arg_keys = inspect.signature(fn).parameters.keys()
    return dict((k, params[k]) for k in arg_keys if k in params)

def run_sim(params, NG,
            c,
            simtime,
            sim_dt=0.1 *ms,
            rv_timestep=500,
            report='text',
            rate_monitors=None,
            state_monitors=None,
            spike_monitors=None,
            recvars=None):
    """
    run a BRIAN2 simulation given the network architecture as NeuronGroups (NG) and connections/synapses (c)
    :param NG: dict of neuron groups with keys == neurons/layers
    :param c: dict of connections / synapses
    :param simtime: duration of simulation
    :param sim_dt: simulation temporal resolution (timestep)
    :param rv_timestep:
    :param report:
    :param rate_monitors: list of neuron group names
    :param state_monitors: list of tuple: (neuron group, variables (tuple), indices (optional))
    :param spike_monitors: list of neuron group names
    :param recvars: creates StateMonitor to record given variables from ALL neuron groups
    :return:
    """

    defaultclock.dt = sim_dt
    net = Network(NG.values(), c.values())

    ### monitors
    if spike_monitors is not None:
        spmons = [SpikeMonitor(NG[mon], record=True) for mon in spike_monitors]
        net.add(spmons)

    if rate_monitors is not None:
        rate_mons = [PopulationRateMonitor(NG[mon], name='rate_{}'.format(mon)) for mon in rate_monitors]
        net.add(rate_mons)

    if recvars is not None:
        var_mons = [StateMonitor(NG[mon], variables=recvars, record=True, dt=rv_timestep) for mon in spike_monitors]
        net.add(var_mons)
    else:
        var_mons = None

    if state_monitors is not None:
        state_mons = [StateMonitor(NG[mon[0]], variables=mon[1], record=(True if len(mon) <= 2 else mon[2]), name='state_{}'.format(mon[0])) for mon in state_monitors]
        net.add(state_mons)

    # RateKC = PopulationRateMonitor(NG['KC'])
    # stateKC = StateMonitor(NG['KC'], 'v', record=True)
    # net.add(stateKC)
    # net.add(RateKC)

    ### run
    net.run(simtime, report=report, namespace=params)

    if spike_monitors is not None:
        out_spmons = dict((spike_monitors[i], sm) for i, sm in enumerate(spmons))
    else:
        out_spmons = None

    # out_spmons.update(dict(('population_' + spike_monitors[i], sm) for i, sm in enumerate(rate_mons)))
    # out_spmons = dict((spike_monitors[i], sm) for i, sm in enumerate(spmons))

    # prepare rate monitors
    if rate_monitors is not None:
        out_pop_mons = dict((rate_monitors[i], sm) for i, sm in enumerate(rate_mons))
    else:
        out_pop_mons = None

    # prepare state monitors
    if state_monitors is not None:
        out_statemons = dict((state_monitors[i][0], sm) for i, sm in enumerate(state_mons))
    else:
        out_statemons = None

    # prepare recvar monitors (this is probably redundant to state_mons ?)
    if var_mons is not None:
        out_var_mons = dict(
            (mon, dict((var, statemon.values) for var, statemon in m.iteritems())) for mon, m in zip(spike_monitors, var_mons))
    else:
        out_var_mons = None

    return out_spmons, out_pop_mons, out_statemons, out_var_mons


def load_sim(filename):
    """
    convenience function to load simulation results from numpy file.
    :param filename:
    :return: AttrDict
    """
    return np.load(filename, allow_pickle=True)['data'][()]

def save_sim(filename, params, spmons, popmons, statemons, simtime, warmup, dt, **kwargs):
    """
    save all results (all monitors, model parameters, ...) from run_sim into a numpy file.
    The monitors will be stored in a pickle-able object with the same attributes as the Brian2 monitors (t,i,spike_trains etc..).
    All time values (Monitors, simtime ...)  are being stored in seconds.
    :param filename:
    :param params:
    :param spmons:
    :param popmons:
    :param statemons:
    :param simtime:
    :param warmup:
    :param dt:
    :param kwargs: custom data to be stored (e.g. stimulus input, tuning profiles ...). All items must be pickle-able.
    :return:
    """

    #SpikeMon = namedtuple('SpikeMonitorLike', ['t', 'i', 'spike_trains'], verbose=False)
    #PopMon = namedtuple('PopulationRateMonitorLike', ['t', 'rate', 'smooth_rate'], verbose=False)

    stateMonData = dict()
    if statemons is not None:
        for k, v in statemons.items():
            # TODO: also store the Quantity / unit ?
            data = {var: v.variables[var].get_value().T for var in v.record_variables}
            data.update({'t': v.t[:] / second})
            stateMonData.update({k: AttrDict(data)})


    data = {
        'spikes': AttrDict({k: AttrDict({'count': v.count[:], 't': v.t[:]/second, 'i': v.i[:], 'spike_trains': v.spike_trains()}) for k,v in spmons.items()}),
        'rates':  AttrDict({k: AttrDict({'t': v.t[:]/second, 'rate': v.rate[:]/Hz, 'smooth_rate': v.smooth_rate(window='flat', width=50*ms)[:] / Hz}) for k,v in popmons.items()}) if popmons is not None else AttrDict({}),
        'variables': AttrDict(stateMonData),
        'simtime': simtime / second,
        'warmup': warmup / second,
        'dt': dt / second,
        'params': params
    }

    data.update(kwargs)

    d = AttrDict(data)
    np.savez(filename, data=d)
    return d

def save_sim_hdf5(filename, params, spmons, popmons, statemons, simtime, warmup, dt, **kwargs):
    """
    save all results (all monitors, model parameters, ...) from run_sim into a HDF5 file.
    The monitors will be stored in a pickle-able object with the same attributes as the Brian2 monitors (t,i,spike_trains etc..).
    All time values (Monitors, simtime ...)  are being stored in seconds.
    :param filename:
    :param params:
    :param spmons:
    :param popmons:
    :param statemons:
    :param simtime:
    :param warmup:
    :param dt:
    :param kwargs: custom data to be stored (e.g. stimulus input, tuning profiles ...). All items must be pickle-able.
    :return:
    """
    import h5py
    #SpikeMon = namedtuple('SpikeMonitorLike', ['t', 'i', 'spike_trains'], verbose=False)
    #PopMon = namedtuple('PopulationRateMonitorLike', ['t', 'rate', 'smooth_rate'], verbose=False)

    def recursively_save_dict_contents_to_group(h5file, path, dic):
        """
        ....
        """
        for key, item in dic.items():
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
                h5file[path + key] = item
            elif isinstance(item, dict):
                recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
            else:
                raise ValueError('Cannot save %s type' % type(item))


    stateMonData = dict()
    for k, v in statemons.items():
        # TODO: also store the Quantity / unit ?
        data = {var: v.variables[var].get_value().T for var in v.record_variables}
        data.update({'t': v.t[:] / second})
        stateMonData.update({k: AttrDict(data)})


    data = {
        'spikes': AttrDict({k: AttrDict({'t': v.t[:]/second, 'i': v.i[:], 'spike_trains': v.spike_trains()}) for k,v in spmons.items()}),
        'rates': AttrDict({k: AttrDict({'t': v.t[:]/second, 'rate': v.rate[:]/Hz, 'smooth_rate': v.smooth_rate(window='flat', width=50*ms)[:] / Hz}) for k,v in popmons.items()}),
        'variables': AttrDict(stateMonData),
        'simtime': simtime / second,
        'warmup': warmup / second,
        'dt': dt / second,
        'params': params
    }

    data.update(kwargs)

    f = h5py.File(filename, "w")
    recursively_save_dict_contents_to_group(f, '/', data)
    f.close()
    return f
    #return np.savez(filename, data=AttrDict(data))

def export_sim_matlab(filename, matFile=None):
    """
    export a simulation file saved with save_sim to MAT file
    :param filename: sim file
    :param matFile: opt. matlab file
    :return:
    """
    import scipy.io as scpio

    data = load_sim(filename)

    sp_trains_aligned = {}
    for k, v in data.spikes.items():
        trial_sp = []
        for s in v.spike_trains.values():
            sp_times = (s / second) - data.warmup
            trial_sp.append(list(sp_times))
        sp_trains_aligned[k] = trial_sp

    spikeData = AttrDict({
        k: AttrDict({'count': v.count[:],
                     't': v.t[:],
                     't_aligned': v.t[:] - data.warmup,
                     'i': v.i[:],
                     'spike_trains': v.spike_trains.values(),
                     'spike_trains_aligned': sp_trains_aligned[k]}) for k, v in data.spikes.items()
    })

    trial_ids = []
    samples = []
    odor_ids = []
    rewards = []
    stim_times = []
    warmup = data.warmup

    trial_sp = []
    for sp in spikeData.KC.spike_trains_aligned:
        sp_times = filter(lambda s: s >= 0.0, sp)  # only spikes AFTER warmup
        trial_sp.append(list(sp_times))

    samples.append(trial_sp)
    stim_times.append(data.stimulus_times)
    rewards.append(data.params['rewards'])
    try:
        odor_ids.append(data.odor_id)
    except AttributeError:
        pass

    output = {
        'trial_ids': trial_ids,
        'targets': rewards,
        'odor_ids': odor_ids,
        'stimulus_times': stim_times,
        'trials': samples,
        'T_trial': data.params['T'],
        'N_trials': len(rewards)
    }

    if matFile is None:
        matFile = "{}.mat".format(filename[:-4])


    scpio.savemat(matFile, {'data': output})
    print("exported {} to: {}", filename, matFile)
    return matFile