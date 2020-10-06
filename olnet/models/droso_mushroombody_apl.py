from brian2 import *
import numpy as np
from olnet import get_args_from_dict


def model_ORN_input(CORN, gLORN, ELORN, tau_IaORN, Ee, tau_syn_e, Ei, tau_syn_i, EIa, VtORN, VrORN, tau_ref, stimulus):
    """
    same ase model_ORN but without adaptation
    :return:
    """
    neuron_eqs = '''
    dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) - g_i*(E_i-v) + I0 + stimulus(t,i))/C_m    : volt (unless refractory) # Ia is the spike triggered adaptation
    dg_e/dt = -g_e/tau_e  : siemens  # post-synaptic exc. conductance # synapses
    dg_i/dt = -g_i/tau_i  : siemens  # post-synaptic inh. conductance
    I0 : amp
    '''

    neuron_modelORN = dict()
    neuron_modelORN['model'] = Equations(neuron_eqs, g_l=gLORN, E_l=ELORN, E_e=Ee, E_i=Ei, E_Ia=EIa, C_m=CORN, tau_e=tau_syn_e,tau_i=tau_syn_i, tau_Ia=tau_IaORN)
    neuron_modelORN['threshold'] = 'v > VtORN'
    neuron_modelORN['reset'] = '''v = VrORN'''  # at reset, membrane v is reset and spike triggered adaptation conductance is increased
    neuron_modelORN['refractory'] = tau_ref

    return neuron_modelORN

def model_ORN(CORN, gLORN, ELORN, tau_IaORN, Ee, tau_syn_e, Ei, tau_syn_i, EIa, VtORN, VrORN, tau_ref, bORN, stimulus):

    neuron_eqs = '''
    dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) - g_i*(E_i-v) - g_Ia*(E_Ia-v) + I0 + stimulus(t,i))/C_m    : volt (unless refractory) # Ia is the spike triggered adaptation
    dg_e/dt = -g_e/tau_e  : siemens  # post-synaptic exc. conductance # synapses
    dg_i/dt = -g_i/tau_i  : siemens  # post-synaptic inh. conductance
    dg_Ia/dt = -g_Ia/tau_Ia : siemens # conductance adaptation 'current'
    I0 : amp
    '''

    neuron_modelORN = dict()
    neuron_modelORN['model'] = Equations(neuron_eqs, g_l=gLORN, E_l=ELORN, E_e=Ee, E_i=Ei, E_Ia=EIa, C_m=CORN, tau_e=tau_syn_e,tau_i=tau_syn_i, tau_Ia=tau_IaORN)
    neuron_modelORN['threshold'] = 'v > VtORN'
    neuron_modelORN['reset'] = '''v = VrORN; g_Ia-=bORN'''  # at reset, membrane v is reset and spike triggered adaptation conductance is increased
    neuron_modelORN['refractory'] = tau_ref

    return neuron_modelORN


def model_PN(CPN, gLPN, ELPN, tau_IaPN, Ee, tau_syn_e, Ei, tau_syn_i, EIa, VtPN, VrPN, tau_ref, bPN):

    neuron_eqs = '''
    dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) - g_i*(E_i-v) - g_Ia*(E_Ia-v) + I0)/C_m    : volt (unless refractory) # Ia is the spike triggered adaptation
    dg_e/dt = -g_e/tau_e  : siemens  # post-synaptic exc. conductance # synapses
    dg_i/dt = -g_i/tau_i  : siemens  # post-synaptic inh. conductance
    dg_Ia/dt = -g_Ia/tau_Ia : siemens # conductance adaptation 'current'
    I0 : amp
    '''

    neuron_modelPN = dict()
    neuron_modelPN['model'] = Equations(neuron_eqs, g_l=gLPN, E_l=ELPN, E_e=Ee, E_i=Ei,E_Ia = EIa, C_m=CPN, tau_e=tau_syn_e, tau_i=tau_syn_i,tau_Ia=tau_IaPN)
    neuron_modelPN['threshold'] = 'v > VtPN'
    neuron_modelPN['reset'] = '''v = VrPN; g_Ia-=bPN'''
    neuron_modelPN['refractory'] = tau_ref

    return neuron_modelPN

def model_LN(CLN, gLLN, ELLN, tau_IaLN, Ee, tau_syn_e, Ei, tau_syn_i, EIa, VtLN, VrLN, tau_ref, bLN):

    neuron_eqs = '''
    dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) - g_i*(E_i-v) - g_Ia*(E_Ia-v) + I0)/C_m    : volt (unless refractory) # Ia is the spike triggered adaptation
    dg_e/dt = -g_e/tau_e  : siemens  # post-synaptic exc. conductance # synapses
    dg_i/dt = -g_i/tau_i  : siemens  # post-synaptic inh. conductance
    dg_Ia/dt = -g_Ia/tau_Ia : siemens # conductance adaptation 'current'
    I0 : amp
    '''

    neuron_modelLN = dict()
    neuron_modelLN['model'] = Equations(neuron_eqs, g_l=gLLN, E_l=ELLN, E_e=Ee, E_i=Ei, E_Ia=EIa, C_m=CLN, tau_e=tau_syn_e, tau_i=tau_syn_i, tau_Ia=tau_IaLN)
    neuron_modelLN['threshold'] = 'v > VtLN'
    neuron_modelLN['reset'] = '''v = VrLN; g_Ia-=bLN'''
    neuron_modelLN['refractory'] = tau_ref

    return neuron_modelLN


def model_KC(CKC, gLKC, ELKC, tau_IaKC, Ee, tau_syn_e, Ei, tau_syn_i, EIa, VtKC, VrKC, tau_ref, bKC):

    neuron_eqs = '''
    dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) - g_i*(E_i-v) - g_Ia*(E_Ia-v) + I0)/C_m    : volt (unless refractory) # Ia is the spike triggered adaptation
    dg_e/dt = -g_e/tau_e  : siemens  # post-synaptic exc. conductance # synapses
    dg_i/dt = -g_i/tau_i  : siemens  # post-synaptic inh. conductance
    dg_Ia/dt = -g_Ia/tau_Ia : siemens # conductance adaptation 'current'
    I0 : amp
    '''

    neuron_modelKC = dict()
    neuron_modelKC['model'] = Equations(neuron_eqs, DeltaT=1 * mV, g_l=gLKC, E_l=ELKC, E_e=Ee, E_i=Ei, E_Ia=EIa, C_m=CKC, tau_e=tau_syn_e,tau_i=tau_syn_i, tau_Ia=tau_IaKC)
    neuron_modelKC['threshold'] = 'v > VtKC'
    neuron_modelKC['reset'] = '''v = VrKC; g_Ia-=bKC'''
    neuron_modelKC['refractory'] = tau_ref

    return neuron_modelKC

def model_APL(CAPL, gLAPL, ELAPL, Ee, tau_syn_e, Ei, tau_syn_i, EIa, VtAPL, VrAPL, tau_ref):

    neuron_eqs = '''
    dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) - g_i*(E_i-v) + I0)/C_m    : volt (unless refractory) # Ia is the spike triggered adaptation
    dg_e/dt = -g_e/tau_e  : siemens  # post-synaptic exc. conductance # synapses
    dg_i/dt = -g_i/tau_i  : siemens  # post-synaptic inh. conductance
    I0 : amp
    '''

    neuron_modelAPL = dict()
    neuron_modelAPL['model'] = Equations(neuron_eqs, g_l=gLAPL, E_l=ELAPL, E_e=Ee, E_i=Ei,E_Ia = EIa, C_m=CAPL, tau_e=tau_syn_e, tau_i=tau_syn_i)
    neuron_modelAPL['threshold'] = 'v > VtAPL'
    neuron_modelAPL['reset'] = '''v = VrAPL'''
    neuron_modelAPL['refractory'] = tau_ref

    return neuron_modelAPL

def network(params,
        input_ng,
        neuron_modelORN,
        neuron_modelPN,
        neuron_modelLN,
        neuron_modelKC,
        neuron_modelAPL,
        wORNinputORN,
        wORNPN,
        wORNLN,
        wLNPN,
        wPNKC,
        wKCAPL,
        wAPLKC,
        N_glu,
        ORNperGlu,
        N_KC,
        PNperKC,
        V0min,
        V0max,
        I0_PN = 0*nA,
        I0_LN = 0*nA,
        I0_KC = 0*nA,
        inh_delay=0 * ms,
        apl_delay=0 * ms):
    '''
    ## ToDo documentation ##
    Connect ORNs to PNs such that ORNperGlu ORNs representing input to one Glu connects to 1 PN
    repeat for every Glu, using connect_full. Connects ORNs to LNs in the same way.
    '''

    #########################     NEURONGROUPS     #########################

    NG = dict()

    # ORN Input
    #n_receptors = ORNperGlu * N_glu

    if input_ng is not None:
        validInputTypes = (PoissonGroup, Group, SpikeSource)
        assert isinstance(input_ng, validInputTypes), "parameter 'input_ng' must be of type: {}".format(validInputTypes)
        NG['ORNinput'] = input_ng

    neuron_params_orn = get_args_from_dict(neuron_modelORN, params)
    neuron_params_pn = get_args_from_dict(neuron_modelPN, params)
    neuron_params_ln = get_args_from_dict(neuron_modelLN, params)
    neuron_params_kc = get_args_from_dict(neuron_modelKC, params)
    neuron_params_apl = get_args_from_dict(neuron_modelAPL, params)

    NG['ORN'] = NeuronGroup(N_glu*ORNperGlu, **neuron_modelORN(**neuron_params_orn), namespace=params, method='euler', name='ORNs')
    NG['ORN'].I0 = I0_PN
    NG['PN'] = NeuronGroup(N_glu, **neuron_modelPN(**neuron_params_pn), namespace=params, method='euler', name='PNs')
    NG['PN'].I0=I0_PN
    NG['LN'] = NeuronGroup(N_glu, **neuron_modelLN(**neuron_params_ln), namespace=params, method='euler', name='LNs')
    NG['LN'].I0=I0_LN
    NG['KC'] = NeuronGroup(N_KC, **neuron_modelKC(**neuron_params_kc), namespace=params, method='euler', name='KCs')
    NG['KC'].I0=I0_KC
    NG['APL'] = NeuronGroup(1, **neuron_modelAPL(**neuron_params_apl), namespace=params, method='euler', name='APL')
    NG['APL'].I0 = 0*nA

    #########################     CONNECTIONS       #########################
    c = dict()

    if input_ng is not None:
        ### input-ORN ###
        c['ORNinputORN'] = Synapses(NG['ORNinput'], NG['ORN'], 'w : siemens', on_pre='g_e+=w', namespace=params)
        for i in np.arange(len(NG['ORN'])):
            #c['ORNinputORN'].connect(i=list(range(i * orn_input_multiplier, (i + 1) * orn_input_multiplier)), j=i)
            c['ORNinputORN'].connect(i=i, j=i)
            c['ORNinputORN'].w = wORNinputORN

    ### ORN-PN ###
    c['ORNPN'] = Synapses(NG['ORN'], NG['PN'], 'w : siemens', on_pre='g_e += w', namespace=params)
    for i in np.arange(N_glu):
        c['ORNPN'].connect(i=list(range(i * ORNperGlu, (i + 1) * ORNperGlu)), j=i)
        c['ORNPN'].w = wORNPN

    ### ORN-LN ###
    c['ORNLN'] = Synapses(NG['ORN'], NG['LN'], 'w : siemens', on_pre='g_e += w', namespace=params)
    for i in np.arange(N_glu):
        c['ORNLN'].connect(i=list(range(i * ORNperGlu, (i + 1) * ORNperGlu)), j=i)
        c['ORNLN'].w = wORNLN

    ### LN-PN ###
    c['LNPN'] = Synapses(NG['LN'], NG['PN'], 'w : siemens', on_pre='g_i -= w', delay=inh_delay, namespace=params)
    c['LNPN'].connect()  # connect_all
    c['LNPN'].w = wLNPN


    ## PN-KC ##
    c['KC'] = Synapses(NG['PN'], NG['KC'], 'w : siemens', on_pre='g_e += w', namespace=params)
    c['KC'].connect(p=PNperKC / float(N_glu))
    c['KC'].w = wPNKC
        # the total number of possible synapses is N_pre*N_post
        # when the connection probability is 0.05 then N_syn = N_pre*N_post*0.05 (on average)
        # every postsynaptic neuron will receive N_syn/N_post synaptic inputs _on average_
        # and every presynaptic input will send out N_syn/N_pre _on average_
        # number of inputs per KC is given by the biominal distribution

    ## KC-APL ##
    c['KCAPL'] = Synapses(NG['KC'], NG['APL'], 'w : siemens', on_pre='g_e += w', delay=apl_delay, namespace=params)
    c['KCAPL'].connect()  # connect_all
    c['KCAPL'].w = wKCAPL

    ## APL-KC ##
    c['APLKC'] = Synapses(NG['APL'], NG['KC'], 'w : siemens', on_pre='g_i -= w', delay=apl_delay, namespace=params)
    c['APLKC'].connect(p=1)
    c['APLKC'].w = wAPLKC

    #########################     INITIAL VALUES     #########################
    #NG['PN'].v = np.random.rand(len(NG['PN']))*(V0max-V0min)+V0min
    #NG['LN'].v = np.random.rand(len(NG['LN']))*(V0max-V0min)+V0min
    #NG['KC'].v = np.random.rand(len(NG['KC']))*(V0max-V0min)+V0min

    NG['ORN'].v = np.random.uniform(V0min, V0max, size=len(NG['ORN'])) * volt
    NG['PN'].v = np.random.uniform(V0min, V0max, size=len(NG['PN'])) * volt
    NG['LN'].v = np.random.uniform(V0min, V0max, size=len(NG['LN'])) * volt
    NG['KC'].v = np.random.uniform(V0min, V0max, size=len(NG['KC'])) * volt
    NG['APL'].v = np.random.uniform(V0min, V0max, size=len(NG['APL'])) * volt

    return NG, c