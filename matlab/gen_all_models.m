n_models = 100;

% classical (lab) cond. on single pulses: target odor 0
[w, train_loss, test_loss, w_init] = msp_fit_mbon_labcond('classicalLabCond-0-15', '../data/LabCond_0-3-5-8-15-3sec.mat', 'n_epochs', 1, 'optimizer', 'rmsprop', 'split', 0.25, 'learn_rate', 0.0005, 'n_models', n_models, 'spikes_per_reward', 1, 'target_odor_id', 0, 'odor_ids', [0,15]);
% target odor: 15
[w, train_loss, test_loss, w_init] = msp_fit_mbon_labcond('classicalLabCond-0-15', '../data/LabCond_0-3-5-8-15-3sec.mat', 'n_epochs', 1, 'optimizer', 'rmsprop', 'split', 0.25, 'learn_rate', 0.0005, 'n_models', n_models, 'spikes_per_reward', 1, 'target_odor_id', 15, 'odor_ids', [0,15]);

% target odor: 0 sparsity 5%
[w, train_loss, test_loss, w_init] = msp_fit_mbon_labcond('classicalLabCondMediumSparsity-0-15', '../data/LabCondMediumSparsity_0-3-5-8-15-3sec.mat', 'n_epochs', 1, 'optimizer', 'rmsprop', 'split', 0.25, 'learn_rate', 0.0005, 'n_models', n_models, 'spikes_per_reward', 1, 'target_odor_id', 0, 'odor_ids', [0,15]);
% target odor: 0 sparsity 10%
[w, train_loss, test_loss, w_init] = msp_fit_mbon_labcond('classicalLabCondLowSparsity-0-15', '../data/LabCondLowSparsity_0-3-5-8-15-3sec.mat', 'n_epochs', 1, 'optimizer', 'rmsprop', 'split', 0.25, 'learn_rate', 0.0005, 'n_models', n_models, 'spikes_per_reward', 1, 'target_odor_id', 0, 'odor_ids', [0,15]);



sequenceDataSets = {
	'../data/PoisonPulse_0-3-15-10sec.mat',
	'../data/PoisonPulse_0-3-5-15-10sec.mat',
	'../data/PoisonPulse_0-3-5-8-15-10sec.mat',
	'../data/PoisonPulse_0-3-8-15-10sec.mat',
	'../data/PoisonPulseMediumSparsity_0-3-8-15-10sec.mat',
	'../data/PoisonPulseLowSparsity_0-3-8-15-10sec.mat'
};

predictionDataSets = {
	'../data/PoisonPulse_0-3-15-10sec.mat',
	'../data/PoisonPulse_0-3-5-15-10sec.mat',
	'../data/PoisonPulse_0-3-5-8-15-10sec.mat',
	'../data/PoisonPulse_0-3-8-15-10sec.mat',
	'../data/Gaussian_15-0-3-15_10sec.mat',
	'../data/PoisonPulseMediumSparsity_0-3-8-15-10sec.mat',
	'../data/PoisonPulseLowSparsity_0-3-8-15-10sec.mat',
	'../data/GaussianMediumSparsity_15-0-3-15_10sec.mat',
	'../data/GaussianLowSparsity_15-0-3-15_10sec.mat'
};

for k=i:length(predictionDataSets)

	if k <= 5
	%%%% predictions for regular sparsiy levels
	% target odor: 0
	[w, train_loss, test_loss, w_init] = msp_fit_mbon_labcond('classicalLabCond-0-15', '../data/LabCond_0-3-5-8-15-3sec.mat', 'n_epochs', 1, 'optimizer', 'rmsprop', 'split', 0.25, 'learn_rate', 0.0005, 'n_models', n_models, 'spikes_per_reward', 1, 'target_odor_id', 0, 'odor_ids', [0,15], 'predictDataSet', predictionDataSets{k});

	% target odor: 15
	[w, train_loss, test_loss, w_init] = msp_fit_mbon_labcond('classicalLabCond-0-15', '../data/LabCond_0-3-5-8-15-3sec.mat', 'n_epochs', 1, 'optimizer', 'rmsprop', 'split', 0.25, 'learn_rate', 0.0005, 'n_models', n_models, 'spikes_per_reward', 1, 'target_odor_id', 15, 'odor_ids', [0,15], 'predictDataSet', predictionDataSets{k});
	else
	%%%% predictions for Low/Medium sparsity levels
	% target odor: 0
	[w, train_loss, test_loss, w_init] = msp_fit_mbon_labcond('classicalLabCondMediumSparsity-0-15', '../data/LabCondMediumSparsity_0-3-5-8-15-3sec.mat', 'n_epochs', 1, 'optimizer', 'rmsprop', 'split', 0.25, 'learn_rate', 0.0005, 'n_models', n_models, 'spikes_per_reward', 1, 'target_odor_id', 0, 'odor_ids', [0,15], 'predictDataSet', predictionDataSets{k});
	end
end

% fit models on sequential task
for m=i:length(sequenceDataSets)
	dataFile = sequenceDataSets{m};
	modelName = dataFile(9:end-4);
	[w, train_loss, test_loss, w_init] = msp_fit_mbon_task(modelName, dataFile, 'n_epochs', 15, 'optimizer', 'rmsprop', 'split', 0.2, 'learn_rate', 0.0005, 'n_models', n_models, 'spikes_per_reward', 1, 'target_odor_id', 0, 'odor_ids', [0,15]);
	% target odor: 15
	[w, train_loss, test_loss, w_init] = msp_fit_mbon_task(modelName, dataFile, 'n_epochs', 15, 'optimizer', 'rmsprop', 'split', 0.2, 'learn_rate', 0.0005, 'n_models', n_models, 'spikes_per_reward', 1, 'target_odor_id', 15, 'odor_ids', [0,15]);

end
