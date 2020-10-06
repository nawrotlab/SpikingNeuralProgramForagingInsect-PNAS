from brian2 import *
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm
import matplotlib.colors as cor
from olnet import load_sim
import numpy as np
import scipy.io as scpio
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import string
#plt.style.use('ggplot')
plt.style.use('figures.mplstyle')
plt.rc('text', usetex=False)
#plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{wasysym}'+
#    r'\usepackage[dvipsnames]{xcolor} \usepackage{MnSymbol}  \usepackage{txfonts}')

def params_as_dict(params):
    layers = ['ORN', 'PN', 'LN', 'KC']
    str_dict = dict({k: dict() for k in layers})
    str_dict['global'] = dict()
    for k in params:
        matched = list(filter(k.endswith, layers))

        if len(matched):
            l_ = matched[0]
            str_dict[l_].update({k[:-len(l_)]: params[k]})
        else:
            str_dict['global'].update({k: params[k]})

    return str_dict

def params_as_latex_table(d):

    header = "\\textbf{Parameter} & \\textbf{Value} & \\textbf{Unit} \\\\ \n"

    print_section = lambda name: "\multicolumn{3}{l}{\\cellcolor[HTML]{343434}{\\color[HTML]{FFFFFF} \\textbf{" + name + "}}} \\\\ \n" + header
    print_row = lambda var,val,unit: "{} & ${:.5f}$ & {} \\\\ \n".format(var.replace('_', '\_'), val, unit if unit is not None else "")
    tab_start = """\\begin{table}[]\n\\centering\n\\begin{tabular}{lll}"""

    tab_end = """\n\end{tabular}\n\end{table}"""

    str = ""

    for sec in d.keys():
        if sec == "global":
            str += tab_start
            str += print_section("Other")
            for k in d[sec].keys():
                if isinstance(d[sec][k], Quantity):
                    v,u = d[sec][k].in_best_unit(5).split(" ")
                    str += print_row(k, float(v), u)
                elif not isinstance(d[sec][k], (list, tuple, np.ndarray)):
                    str += print_row(k, d[sec][k], None)
                else:
                    pass
            str = str[:-4]
            str += tab_end
        else:
            str += tab_start
            str += print_section(sec)
            for k in d[sec].keys():
                if isinstance(d[sec][k], Quantity):
                    v,u = d[sec][k].in_best_unit(5).split(" ")
                    str += print_row(k, float(v), u)
                elif isinstance(d[sec][k], (list, tuple, np.ndarray)):
                    pass
                else:
                    str += print_row(k, d[sec][k], None)
            str = str[:-4]
            str += tab_end


    return str

def legendAsLatex(axes, rotation=90) :
    '''Generate a latex code to be used instead of the legend.
       Uses the label, color, marker and linestyle provided to the pyplot.plot.
       The marker and the linestyle must be defined using the one or two character
           abreviations shown in the help of pyplot.plot.
       Rotation of the markers must be multiple of 90.
    '''
    latexLine = {'-':'\\textbf{\Large ---}',
        '-.':'\\textbf{\Large --\:\!$\\boldsymbol{\cdot}$\:\!--}',
        '--':'\\textbf{\Large --\,--}',':':'\\textbf{\Large -\:\!-}'}
    latexSymbol = {'o':'medbullet', 'd':'diamond', 's':'filledmedsquare',
        'D':'Diamondblack', '*':'bigstar', '+':'boldsymbol{\plus}',
        'x':'boldsymbol{\\times}', 'p':'pentagon', 'h':'hexagon',
        ',':'boldsymbol{\cdot}', '_':'boldsymbol{\minus}','<':'LHD',
        '>':'RHD','v':'blacktriangledown', '^':'blacktriangle'}
    rot90=['^','<','v','>']
    di = [0,-1,2,1][rotation%360//90]
    latexSymbol.update({rot90[i]:latexSymbol[rot90[(i+di)%4]] for i in range(4)})
    return ', '.join(['\\textcolor[rgb]{'\
            + ','.join([str(x) for x in cor.to_rgb(handle.get_color())]) +'}{'
            + '$\\'+latexSymbol.get(handle.get_marker(),';')+'$'
            + latexLine.get(handle.get_linestyle(),'') + '} ' + label
                for handle,label in zip(*axes.get_legend_handles_labels())])

def figure1_mst(file, modelIdx=3, odorIdx=0, modelType='msp_classicalLabCondLowSparsity-0-15.odor-0.1-sp', **args):
    dataSetname = file[6:file[6:].index('/') + 6]
    mstFile = "matlab/model_cache/predictions/{}.{}/{}.mat".format(modelType, modelIdx, dataSetname)
    trialIdx = int(file.split('/')[-1].split('-')[1])
    print("trialIdx: {} | mstFile: {}".format(trialIdx, mstFile))
    data = load_sim(file)
    return figure1(data, mstMatFile=mstFile, mstTrialIdx=trialIdx, mstOdorIdx=odorIdx, **args)


def figure1(data, show_rate=False, t_min=0, t_max=None, orn_range=None, pn_range=None, ln_range=None, kc_range=None, cmap=None, mstMatFile=None, mstOdorIdx=None, mstTrialIdx=None, fig_size=None):

    if type(data) == str:
        data = load_sim(data)

    dt = data.dt
    simtime = data.simtime
    warmup_time= data.warmup
    M = data.tuning
    stimulus = data.stimulus
    spikemons = data.spikes
    pop_mons = data.rates
    state_mons = data.variables

    t_max = data.simtime - data.warmup if t_max is None else t_max
    t_min = 0 if t_min is None else t_min
    col_map = 'Reds' if cmap is None else cmap
    t_offset = 0
    tempotron_sp = None

    if mstMatFile is not None:
        mat = scpio.loadmat(mstMatFile)
        tempotron_sp = mat['sp_times'][0, mstTrialIdx]
        print("MST spikes #{}: {}".format(len(tempotron_sp), tempotron_sp))
        print("stimulus times odor #{} #{}: {}".format(mstOdorIdx, len(data.stimulus_times[mstOdorIdx]), data.stimulus_times[mstOdorIdx]))
        if len(data.stimulus_times[mstOdorIdx]) == 1:
            t_offset = data.stimulus_times[mstOdorIdx][0]
            print("stimulus onset: {}".format(t_offset))

    #orn_range = [1, np.max(spikemons['ORN'].i) + 1] if orn_range is None else orn_range
    #pn_range = [1, np.max(spikemons['PN'].i) + 1] if pn_range is None else pn_range
    #ln_range = [1, np.max(spikemons['LN'].i) + 1] if ln_range is None else ln_range
    #kc_range = [1, np.max(spikemons['KC'].i) + 1] if kc_range is None else kc_range

    print("orn_range: {}".format(orn_range))
    # trunacte stimulus array to t_min,t_max
    stim_dt = data.simtime / data.stimulus.shape[1]
    stim_from = int((data.warmup + t_min) / stim_dt)
    stim_to = int((data.warmup + t_max) / stim_dt)
    stimulus = data.stimulus[:, stim_from:stim_to]


    smons = [
            ('ORN', ('v', 'g_i', 'g_e'), [360]),
            ('PN', ('v', 'g_i', 'g_e'), [15]),
            ('LN', ('v', 'g_i', 'g_e'), [15])
    ]

    axs = []
    connectivity_axs = []

    fig = plt.figure(num=1, figsize=(4, 6) if fig_size is None else fig_size)
    n_ticks_timeaxis = 10
    y_label_pad = -0.01

    rate_win = 50 * ms
    # KC_y_lim = 100

    outer_grid = gridspec.GridSpec(2, 2, hspace=0.0, wspace=0.01, height_ratios=[.1, .9], width_ratios=[.9, .1])

    network_cell = outer_grid[1:, :-1]
    if tempotron_sp is None:
        gs_spikes = gridspec.GridSpecFromSubplotSpec(7, 1, network_cell, hspace=0.0,
                                                     wspace=0.0, height_ratios=[.1, .2, .1, .3, .1, .1, .1])
    else:
        gs_spikes = gridspec.GridSpecFromSubplotSpec(8, 1, network_cell, hspace=0.0,
                                                     wspace=0.0, height_ratios=[.1, .1, .1, .3, .1, .1, .1, .1])

    # gs_connectivity = gridspec.GridSpecFromSubplotSpec(3, 3, outer_grid[4:, -1])

    # stimulus
    axs.append(plt.subplot(outer_grid[0:1, :-1]))
    # ORN tuning
    #divider = make_axes_locatable(axs[0])
    #ax_tuning = divider.append_axes("right", 1, pad=0.0)
    #axs.append(ax_tuning)

    # ORN tuning
    axs.append(plt.subplot(outer_grid[0:1, -1])) #1

    # network spiking activity
    if orn_range == -1:
        axs.append(plt.subplot(gs_spikes[0:2, :]))      # 2 no ORNs - only plot PNs !
    else:
        axs.append(plt.subplot(gs_spikes[0, :]))                    # 2 ORNs
        axs.append(plt.subplot(gs_spikes[1, :])) #sharex=axs[2]     # 3 PNs

    axs.append(plt.subplot(gs_spikes[2, :])) #sharex=axs[3]     # LNs

    if tempotron_sp is not None:
        axs.append(plt.subplot(gs_spikes[3:-3, :]))  # KCs
        axs.append(plt.subplot(gs_spikes[-3, :]))    # KC histogram
        axs.append(plt.subplot(gs_spikes[-2, :]))  # APL
        axs.append(plt.subplot(gs_spikes[-1, :]))   #  MST output
    else:
        axs.append(plt.subplot(gs_spikes[3:-2, :]))  # KCs
        axs.append(plt.subplot(gs_spikes[-2, :]))  # KC histogram
        axs.append(plt.subplot(gs_spikes[-1, :]))  # APL


    # axis handles
    if orn_range == -1:
        ax_orn = None
        ax_pn, ax_ln, ax_kc, ax_kc_hist = (2, 3, 4, 5)
    else:
        ax_orn, ax_pn, ax_ln, ax_kc, ax_kc_hist = (2, 3, 4, 5, 6)

    ax_apl = -1
    #ax_orn_hist, ax_pn_hist, ax_ln_hist, ax_kc_hist = (6, 7, 8, 9)

    # ORN tuning plot
    chars = string.ascii_uppercase #['A', 'B']
    xs = np.arange(M.shape[1])
    for i, p in enumerate(range(M.shape[0])):
        axs[1].plot(M[i, :, 0], xs, color='C{}'.format(8 - i), label='{}'.format(chars[i]))
    #axs[1].yaxis.set_major_locator(MaxNLocator(4, integer=True))
    axs[1].set_ylabel('')
    axs[1].set_yticks([])
    axs[1].set_xlabel('')
    axs[1].set_xticks([])
    #axs[1].set_ylabel(legendAsLatex(axs[1]))
    #axs[1].set_xlabel('sensivity [a.u.]')
    #axs[1].xaxis.set_label_position('top')
    #axs[1].xaxis.set_ticks_position('top')
    leg = axs[1].legend(loc='upper left', frameon=False, markerscale=.2)
    plt.setp(leg.get_texts(), fontsize='8')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['left'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)

    # stimulus plot
    coords = (t_min, t_max, 1, M.shape[1])
    h = axs[0].imshow(stimulus, interpolation='hamming', aspect='auto', extent=coords, cmap=col_map)
    #divider = make_axes_locatable(axs[0])
    #cax = divider.append_axes("top", size="5%", pad=0.05)
    #cb = fig.colorbar(h, cax=cax, orientation="horizontal", format=plt.NullFormatter())
    #cb.set_label('intensity [a.u.]')
    #cax.xaxis.set_ticks_position("top")
    #cax.xaxis.set_label_position("top")
    #axs[0].yaxis.set_major_locator(MaxNLocator(4, integer=True))
    #axs[0].xaxis.set_major_locator(MaxNLocator(n_ticks_timeaxis, integer=True))
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    #axs[0].set_yticks([stimulus.shape[0]])
    axs[0].set_ylabel("ORNs\n")
    axs[0].yaxis.set_label_coords(y_label_pad, .5)
    axs[0].spines['bottom'].set_visible(False)
    # axs[0].set_xlabel('time [sec]')
    # axs[0].set_title('ORN stimulation input')



    # ORN rasterplot
    if ax_orn is not None:
        axs[ax_orn].plot(spikemons['ORN'].t - warmup_time - t_offset, spikemons['ORN'].i + 1, '|', linewidth=0.5, markersize=1,
                         color='C1')
        axs[ax_orn].set_xlim(t_min - t_offset, t_max - t_offset)
        # axs[ax_orn].yaxis.set_major_locator(MaxNLocator(2, integer=True))
        # axs[ax_orn].xaxis.set_major_locator(MaxNLocator(n_ticks_timeaxis, integer=True))
        axs[ax_orn].set_ylabel("ORNs\n({})".format(orn_range[-1] - orn_range[0] if orn_range is not None else len(spikemons['ORN'].count[:])))
        axs[ax_orn].set_xticks([])
        if orn_range:
            axs[ax_orn].set_ylim(orn_range)
        #axs[ax_orn].set_yticks([orn_range[-1] - orn_range[0]])
        axs[ax_orn].set_yticks([])
        axs[ax_orn].yaxis.set_label_coords(y_label_pad, .5)
        axs[ax_orn].spines['bottom'].set_visible(False)
        # pop rate
        if (show_rate):
            ax_rate = axs[ax_orn].twinx()
            ax_rate.plot((pop_mons['ORN'].t - warmup_time),
                         pop_mons['ORN'].smooth_rate, color='k', alpha=0.8)
            mean_rate = pop_mons['ORN'].smooth_rate[int(warmup_time / dt):].mean()
            ax_rate.set_yticks([mean_rate])


    # PN rasterplot
    axs[ax_pn].plot(spikemons['PN'].t - warmup_time - t_offset, spikemons['PN'].i + 1, '|', linewidth=0.5, markersize=1,
                    color='C0')
    axs[ax_pn].set_xlim(t_min - t_offset, t_max - t_offset)
    axs[ax_pn].set_xticks([])
    axs[ax_pn].set_ylabel("PNs\n({})".format(pn_range[-1] - pn_range[0] if pn_range is not None else len(spikemons['PN'].count[:])))
    if pn_range:
        axs[ax_pn].set_ylim(pn_range)
    #axs[ax_pn].set_yticks([pn_range[-1] - pn_range[0]])
    axs[ax_pn].set_yticks([])
    axs[ax_pn].yaxis.set_label_coords(y_label_pad, .5)
    axs[ax_pn].spines['bottom'].set_visible(False)
    # pop rate
    if (show_rate):
        ax_rate = axs[ax_pn].twinx()
        ax_rate.plot((pop_mons['PN'].t - warmup_time),
                     pop_mons['PN'].smooth_rate, color='k', alpha=0.8)
        mean_rate = pop_mons['PN'].smooth_rate[int(warmup_time/dt):].mean()
        ax_rate.set_yticks([mean_rate])

    # LN rasterplot
    axs[ax_ln].plot(spikemons['LN'].t - warmup_time - t_offset, spikemons['LN'].i + 1, '|', linewidth=0.5, markersize=1,
                    color='C2')
    axs[ax_ln].set_xlim(t_min - t_offset, t_max - t_offset)
    axs[ax_ln].set_xticks([])
    axs[ax_ln].set_ylabel("LNs\n({})".format(ln_range[-1] - ln_range[0] if ln_range is not None else len(spikemons['LN'].count[:])))
    if ln_range:
        axs[ax_ln].set_ylim(ln_range)
    #axs[ax_ln].set_yticks([ln_range[-1] - ln_range[0]])
    axs[ax_ln].set_yticks([])
    axs[ax_ln].yaxis.set_label_coords(y_label_pad, .5)
    axs[ax_ln].spines['bottom'].set_visible(False)
    # pop. rate
    if (show_rate):
        ax_rate = axs[ax_ln].twinx()
        ax_rate.plot((pop_mons['LN'].t - warmup_time),
                     pop_mons['LN'].smooth_rate, color='k', alpha=0.8)
        mean_rate = pop_mons['LN'].smooth_rate[int(warmup_time / dt):].mean()
        ax_rate.set_yticks([mean_rate])

    # KC rasterplot
    print((spikemons['KC'].i))
    print((spikemons['KC'].t - warmup_time - t_offset))
    axs[ax_kc].plot(spikemons['KC'].t - warmup_time - t_offset, spikemons['KC'].i + 1, '|', linewidth=0.5, markersize=1,
                    color='C3')
    #axs[ax_kc].set_ylim(1, len(spikemons['KC'].count[:]))
    axs[ax_kc].set_xlim(t_min - t_offset, t_max - t_offset)
    axs[ax_kc].set_xticks([])
    axs[ax_kc].yaxis.set_major_locator(MaxNLocator(2, integer=True))
    #axs[ax_kc].xaxis.set_major_locator(MaxNLocator(n_ticks_timeaxis, integer=True))
    axs[ax_kc].set_ylabel("KCs\n({})".format(kc_range[-1] - kc_range[0] if kc_range is not None else len(spikemons['KC'].count[:])))
    if kc_range:
        axs[ax_kc].set_ylim(kc_range)
    else:
        axs[ax_kc].set_ylim(1, len(spikemons['KC'].count[:]))
    #axs[ax_kc].set_yticks([kc_range[-1] - kc_range[0]])
    axs[ax_kc].set_yticks([])
    axs[ax_kc].yaxis.set_label_coords(y_label_pad, .5)

    # pop. rate
    if (show_rate):
        ax_rate = axs[ax_kc].twinx()
        ax_rate.set_ylabel('pop. rate [Hz]')
        ax_rate.yaxis.set_label_coords(1.01, .5)
        ax_rate.plot((pop_mons['KC'].t - warmup_time),
                     pop_mons['KC'].smooth_rate, color='k', alpha=0.8)
        mean_rate = pop_mons['KC'].smooth_rate[int(warmup_time / dt):].mean()
        ax_rate.set_yticks([mean_rate])


    ts = np.arange(0, simtime, dt)
    n_kcs = len(spikemons['KC'].count[:])
    bins = np.arange(0, len(spikemons['KC'].count[:]) + 1)
    print("#KCs={}".format(len(spikemons['KC'].count[:])))
    kc_active_percentage = [np.where(np.logical_and(spikemons['KC'].t >= t, spikemons['KC'].t <= t + dt)) for t in ts]

    for i,idx in enumerate(kc_active_percentage):
        kc_active_percentage[i] = (len(np.unique(spikemons['KC'].i[idx])) * 100) / n_kcs

    #y = [1 - ((x.mean()**2) / ((x*x).mean()+ 1e-8)) for x in tmp]
    #print(kc_active_percentage)
    idx = np.where((ts - warmup_time) >= t_min)[0].tolist()
    axs[ax_kc_hist].plot(ts - warmup_time - t_offset, kc_active_percentage, 'k', drawstyle='steps')
    axs[ax_kc_hist].fill_between(ts - warmup_time - t_offset, kc_active_percentage, color='k', step="pre")
    axs[ax_kc_hist].set_ylabel("% KCs\n")
    axs[ax_kc_hist].yaxis.set_label_coords(y_label_pad, .5)
    axs[ax_kc_hist].yaxis.set_tick_params({'pad': 0.05})
    #axs[ax_kc_hist].yaxis.set_major_locator(MaxNLocator(2, integer=True))
    axs[ax_kc_hist].set_xlim(t_min - t_offset, t_max - t_offset)
    axs[ax_kc_hist].set_ylim(0, np.max(kc_active_percentage[idx[0]:]) + 0.5)
    axs[ax_kc_hist].set_xticks([])
    axs[ax_kc_hist].set_yticks([0, np.ceil(np.max(kc_active_percentage[idx[0]:]))])
    #ax_rate.plot(ts - warmup_time, y, 'k', alpha=0.8)
    #ax_rate.set_ylim(0.8,1.1)
    #ax_rate.set_ylabel('sparseness')
    # axs[ax_kc].set_xlabel('time [sec]')
    #axs[ax_kc].set_yticks(
    #    np.arange(start=1, step=int(len(spikemons['KC'].count[:]) // 2) - 1, stop=len(spikemons['KC'].count[:])))

    if tempotron_sp is not None:
        ax_apl = -2
        #axs[-1].vlines(np.array(data.stimulus_times[mstOdorIdx]) - t_offset, 1.5, 2.5, linewidth=1.5, color=[.3,.3,.3])
        axs[-1].vlines(np.array(tempotron_sp) - t_offset, 0, 1, linewidth=1.5, color=[.3,.3,.3])
        axs[-1].set_yticks([])
        axs[-1].set_ylim(-.1, 1.5)
        axs[ax_kc_hist].yaxis.set_label_coords(y_label_pad, .5)
        axs[-1].yaxis.set_tick_params({'pad': 0.05})
        axs[-1].set_ylabel("MBON\n")

    # APL rasterplot
    if 'APL' in spikemons.keys():
        #print(spikemons['APL'].t - warmup_time - t_offset)
        axs[ax_apl].plot(spikemons['APL'].t - warmup_time - t_offset, spikemons['APL'].i + 1, '|', linewidth=.5, markersize=10,
                        color='k')
        axs[ax_apl].set_xlim(t_min - t_offset, t_max - t_offset)
        axs[ax_apl].set_xticks([])
        axs[ax_apl].yaxis.set_major_locator(MaxNLocator(2, integer=True))
        axs[ax_apl].set_ylabel("APL\n")
        axs[ax_apl].set_ylim([0.9, 1.2])
        #axs[ax_apl].set_ylim(1)
        axs[ax_apl].set_yticks([])
        axs[ax_apl].yaxis.set_tick_params({'pad': 0.05})
        axs[ax_apl].yaxis.set_label_coords(y_label_pad, .5)

    # time axis on last subplot
    axs[-1].set_xlim(t_min - t_offset, t_max - t_offset)
    axs[-1].xaxis.set_major_locator(MaxNLocator(5, integer=True))
    axs[-1].set_xlabel('time [sec]')

    fig.align_labels(axs[ax_kc])

    #fig.tight_layout(w_pad=0.5, h_pad=0.5)
    return fig


def tempotron_response(matFile, mstOdorIdx, ax=None, modelName=None, showGaussian=False, t_min=0, t_max=10):

    fig = None

    if ax is None:
        print("creating new figure ...")
        fig = plt.figure(figsize=(8,2))
        ax = plt.gca()


    def get_tempotron_spikes(matFile, modelName=None):
        mat = scpio.loadmat(matFile)
        tempotron_sp = None
        N_models = len(mat['data']['predictions'][0][0][0])
        if modelName is not None:
            for k in range(N_models):
                row = mat['data']['predictions'][0][0][0][k]
                if modelName == row[0][0][0]:
                    tempotron_sp = row[0][5][0][0][0]

            if tempotron_sp is None:
                raise Exception("model {} not found".format(modelName))

            return tempotron_sp
        else:
            # use the first available prediction
            return mat['data']['predictions'][0][0][0][0][0][5][0][0][0]

    spikes_y_offset = .8
    tick_size = .5

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    #dt = 1/10000
    #t = np.arange(0, kernel_duration, dt)
    #x, _ = np.histogram(tempotron_sp, bins=np.arange(t_min, t_max + 2 * kernel_duration + dt, dt))
    #rate_est = np.convolve(x, norm_kernel, 'same')[len(t):len(t) + 1]

    for k,f in enumerate(matFile):
        data = load_sim(f[:-4] + ".npz")
        tempotron_sp = get_tempotron_spikes(f, modelName)

        ax.vlines(tempotron_sp + (k*t_max), spikes_y_offset, spikes_y_offset + tick_size, linewidth=1., color='r', label='output spike')
        ax.vlines(np.array(data.stimulus_times[mstOdorIdx]) + (k*t_max), 0, tick_size, linewidth=1., color='b', label='filament crossing')

        if showGaussian:
            xs = np.linspace(t_min, t_max, 10000)
            ax2 = ax.twinx()
            pdf_g = stats.norm.pdf(xs, showGaussian[0], showGaussian[1])
            ax2.plot(xs + (k*t_max), pdf_g + spikes_y_offset, color='k', alpha=.6, linewidth=2., label="Norm({},{})".format(showGaussian[0], showGaussian[1]))
            ax2.fill_between(xs + (k*t_max), pdf_g + spikes_y_offset, facecolor='b', alpha=0.1)
            ax2.set_yticks([])
            ax2.set_ylim(spikes_y_offset, spikes_y_offset + tick_size)
            ax2.set_ylabel(None)
            ax2.spines['top'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)

    ax2.legend(['Pr(filament)'], loc='upper right')

    ax.set_yticks([])
    ax.set_ylim(0, (spikes_y_offset + tick_size) * 1.2)
    ax.set_xlim(t_min, len(matFile) * t_max)
    ax.set_xlabel('time [sec]')
    ax.legend(['filament crossing', 'output spike'], loc='upper left', ncol=2)

    if (fig is not None):
        fig.tight_layout()

    return fig
