from olnet.plotting.figures import figure1
import numpy as np
import matplotlib.pyplot as plt
fileType = "png"

# plot LabConditioning single-trial
file = 'cache/LabCond_0-3-5-8-15-3sec/sim-odor-0-0-58.npz'
mstMATFile = 'matlab/model_cache/predictions/msp_classicalLabCond-0-15.odor-0.1-sp.1/LabCond_0-3-5-8-15-3sec.mat'
data = np.load(file)['data'][()]
figure_1 = figure1(data, t_min=1.0, t_max=1.3, orn_range=[620,680], pn_range=[0,35], cmap='seismic',
                   mstMatFile=mstMATFile, mstOdorIdx=0, mstTrialIdx=0, fig_size=(3.5, 6))
figure_1.savefig("figures/system_response_labcond.{}".format(fileType), dpi=300)

file = 'cache/PoisonPulse_0-3-5-8-15-10sec/sim-12-90.npz'
mstMATFile = 'matlab/model_cache/predictions/msp_classicalLabCond-0-15.odor-0.1-sp.1/PoisonPulse_0-3-5-8-15-10sec.mat'
data = np.load(file)['data'][()]
figure_2 = figure1(data, t_max=8, orn_range=[640,665], pn_range=[0,25], cmap='seismic',mstMatFile=mstMATFile, mstOdorIdx=0, mstTrialIdx=12)
figure_2.savefig("figures/system_response_poisson.{}".format(fileType), dpi=300)

figure_2_alt = figure1(data, t_max=8, orn_range=-1, pn_range=[0,45], cmap='seismic',mstMatFile=mstMATFile, mstOdorIdx=0, mstTrialIdx=12)
figure_2_alt.savefig("figures/system_response_poisson_noORN.{}".format(fileType), dpi=300, fig_size=(4.25,8))

file = 'cache/GaussianCone_15-0-3-15_10sec/sim-13-27.npz'
mstMATFile = 'matlab/model_cache/predictions/msp_classicalLabCond-0-15.odor-15.1-sp.1/Gaussian_15-0-3-15_10sec.mat'
data = np.load(file)['data'][()]
figure_3 = figure1(data, t_max=10, orn_range=-1, pn_range=[0,35], cmap='seismic',mstMatFile=mstMATFile, mstOdorIdx=15, mstTrialIdx=13)
figure_3.savefig("figures/system_response_gaussian.{}".format(fileType), dpi=300)

figure_3_alt = figure1(data, t_max=10, orn_range=-1, pn_range=[0,35], cmap='seismic')
figure_3_alt.savefig("figures/system_response_gaussian_noMST.{}".format(fileType), dpi=300)