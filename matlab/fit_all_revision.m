target_odor_id = 15;
n_models = 50;
n_train_samples = -1;

train_instances = {
    'ConnectivityMediumSparsityAPL',
    'ConnectivityLowSparsityAPL',
    'ConnectivityHighSparsityAPL'
    %'WeightMediumSparsityAPL'
};

data_sets = {
    'Gaussian%s_15-0-3-15_10sec.mat',
    'PoisonPulse%s_0-3-15-10sec.mat',
    'PoisonPulse%s_0-3-5-15-10sec.mat',
    'PoisonPulse%s_0-3-5-8-15-10sec.mat',
    'PoisonPulse%s_0-3-8-15-10sec.mat'
};

for i=1:length(train_instances)
    variant = train_instances{i};
    [w, train_loss, test_loss, w_init] = msp_fit_mbon_labcond(['classicalLabCond', variant, '-0-15'], ...
        ['../data/LabCond', variant, '_0-15-3sec.mat'], ...
        'n_samples', n_train_samples, 'n_epochs', 1, 'optimizer', 'rmsprop', 'split', 0.25, 'learn_rate', 0.0005, ...
        'n_models', n_models, 'spikes_per_reward', 1, 'target_odor_id', target_odor_id, 'odor_ids', [0,15]);
end


for i=1:length(train_instances)
    variant = train_instances{i};
    for j=1:length(data_sets)
        dataSet = data_sets{j};
        [w, train_loss, test_loss, w_init] = msp_fit_mbon_labcond(['classicalLabCond', variant, '-0-15'], ...
        ['../data/LabCond', variant, '_0-15-3sec.mat'], ...
        'n_epochs', 1, 'optimizer', 'rmsprop', 'split', 0.25, 'learn_rate', 0.0005, ...
        'n_models', n_models, 'spikes_per_reward', 1, 'target_odor_id', target_odor_id, 'odor_ids', [0,15], ...
        'predictDataSet', sprintf(['../data/', dataSet], variant), 'predictWeightIdx', -1); 
    end
end