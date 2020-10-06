function [w_out, train_losses, validation_losses, w_init] = msp_fit_mbon_task(modelName, dataSetFileName, varargin)

args = inputParser;
defaultOptimizer='rmsprop';
validOptimizers = {'rmsprop', 'momentum'};
checkOptimizer = @(x) any(validatestring(x,validOptimizers));
defaultCvMethod = 'traintest';
validCvMethods = {'traintest', 'kfold'};
checkCvMethod = @(x) any(validatestring(x,validCvMethods));

addRequired(args,'modelName',@ischar);
addRequired(args,'dataSetFileName',@ischar);
addParameter(args, 'odor_ids', []);
addParameter(args, 'n_samples', -1, @isnumeric);
addParameter(args, 'n_epochs', 10, @isnumeric);
addParameter(args, 'dt', 1/1000, @isnumeric);
addParameter(args, 'optimizer', defaultOptimizer, checkOptimizer);
addParameter(args, 'rng_seed', 42, @isnumeric);
addParameter(args, 'split', 0.3, @isnumeric);
addParameter(args, 'cv_method', defaultCvMethod, checkCvMethod);
addParameter(args, 'learn_rate', 0.001, @isnumeric);
addParameter(args, 'spikes_per_reward', 1, @isnumeric);
addParameter(args, 'target_odor_id', 2, @isnumeric);
addParameter(args, 'early_stopping', -1, @isnumeric);
addParameter(args, 'n_models', 15, @isnumeric);

args.KeepUnmatched = true;
parse(args,modelName, dataSetFileName, varargin{:});

n_samples = args.Results.n_samples;
n_models = args.Results.n_models;
n_epochs = args.Results.n_epochs;
odor_ids = args.Results.odor_ids;
optimizer = args.Results.optimizer;
seed = args.Results.rng_seed;
train_test_split = args.Results.split;
cvMethod = args.Results.cv_method;
early_stopping_accuracy = args.Results.early_stopping;
seeds = randi(98756, 1, n_models);

f = load(dataSetFileName);
data = f.data;

N_syn = size(data.trials, 2);
dt = args.Results.dt;
T = double(data.T_trial);
lr = args.Results.learn_rate;
ts = 0:dt:T;
reward_size = args.Results.spikes_per_reward; % no of spikes for individual pattern
target_idx = args.Results.target_odor_id + 1; % idx of rewarded odor source (data.rewards contains multiple counts for each odor)

% neuron model
tau_m = 0.015;
tau_s = 0.005;
V_thresh = 1;
V_rest = 0;
rng(seed);

if isfield(data, 'odor_ids') && ~isempty(odor_ids)
    disp(['filtering for odor_ids: ' num2str(odor_ids)]);
    idx = ismember(data.odor_ids, odor_ids);
    data.trials = data.trials(idx, :);
    data.targets = data.targets(idx, :);
    data.odor_ids = data.odor_ids(idx);
end

y = double(data.targets);
y(:, target_idx) = y(:, target_idx) .* reward_size;
y = y(:,target_idx);

if n_samples > 0   
   %c = cvpartition(y,'HoldOut', size(data.trials, 2) - n_samples) ;
   rnd_idx = randperm(size(data.trials,1));
   
   % shuffle
   data.trials = data.trials(rnd_idx,:);
   data.targets = data.targets(rnd_idx, :);
   % re-partition
   data.trials = data.trials(1:n_samples,:);
   data.targets = data.targets(1:n_samples, :);
   
   if isfield(data, 'odor_ids')
      data.odor_ids = data.odor_ids(rnd_idx); 
      data.odor_ids = data.odor_ids(1:n_samples); 
   end
   
   y = y(rnd_idx);
   y = y(1:n_samples);
   disp(sprintf('adjusted dataSet size: %d | rewards: %d', size(data.trials, 1), length(data.targets)));

end

disp(sprintf('n_samples: %d | optimizer: %s | n_epochs: %d | cvMethod: %s | reward_size: %d | target_idx: %d', n_samples, optimizer, n_epochs, cvMethod, reward_size, target_idx));

 
if isfield(data, 'odor_ids')
    % partition data - stratified HoldOut
    if strcmpi(cvMethod, 'kfold') == 1
        cval = cvpartition(y,'KFold', train_test_split);
    else
        cval = cvpartition(y, 'HoldOut', train_test_split);
    end
else
    % partition data
    if strcmpi(cvMethod, 'kfold') == 1
        cval = cvpartition(y,'KFold', train_test_split);
    else
        cval = cvpartition(size(data.trials,1), 'HoldOut', train_test_split);
    end
end


n_folds = 1;

if strcmpi(cvMethod, 'kfold') == 1
    n_folds = cval.NumTestSets;
    cval
end


for m=1:n_models

model_seed = seeds(m);
rng(model_seed);
outFile = sprintf('model_cache/msp_%s.odor-%d.%d-sp.%d.mat', modelName, args.Results.target_odor_id, args.Results.spikes_per_reward, m);
c = repartition(cval);
%w_outs = zeros(n_folds, N_syn);
%w_inits = zeros(n_folds, N_syn);
w_inits = normrnd(0, 1 / N_syn, n_folds, N_syn);
w_outs = w_inits;

train_losses = zeros(n_folds, n_epochs);
train_accuracy = zeros(n_folds, n_epochs);
validation_losses = zeros(n_folds, n_epochs);    
validation_accuracy = zeros(n_folds, n_epochs);    
   
for k=1:n_folds
    n_iter = 0;
    w_out = w_outs(k, :);
    w_init = w_inits(k, :);
    
    % split train data
    train_data = data.trials(c.training(k), :);
    y_train = y(c.training(k))';
    n_train = length(y_train);
    % shuffle train set labels
    shuffle_idx_train = randperm(length(y_train));
    train_data = train_data(shuffle_idx_train, :);
    y_train = y_train(shuffle_idx_train);
    
    % split test/validation data
    test_data = data.trials(c.test(k), :);
    y_test = y(c.test(k))';
    n_test = length(y_test(k));

    for i=1:n_epochs
            [w_out, ~, ~, errs, preds, ~, ~, n_iter] = fit_msp_tempotron(ts, train_data, y_train, w_out, V_thresh, V_rest, tau_m, tau_s, lr, n_iter, optimizer);
            w_outs(k,:) = w_out;
            loss = mean(abs(preds-y_train));
            n_correct = length(find(preds==y_train));
            train_accuracy(k,i) = (n_correct * 100)/length(y_train);
            train_losses(k,i) = loss;
            
            [mean_val_loss, ~, val_preds, ~] = validate_msp_tempotron(ts, test_data, y_test, w_out, V_thresh, V_rest, tau_m, tau_s);
            validation_losses(k,i) = mean_val_loss;
            n_correct = length(find(val_preds==y_test));
            n_total = length(y_test);
            validation_accuracy(k,i) = (n_correct * 100)/n_total;

            if early_stopping_accuracy > 0 && validation_accuracy(1,i) >= early_stopping_accuracy
                disp(sprintf('[%d] early stopping @ %.3f | %s learning converged after %d epochs val_accuracy: %.3f', k, early_stopping_accuracy, optimizer, i, validation_accuracy(1,i)));
                break;
            end

            if (isempty(errs)) % all zeros => no errors, converged
                disp(sprintf('[%d] %s learning converged after %d epochs', k, optimizer, i));
                break;
            end

            if (mod(i, 1) == 0)
                disp(sprintf('[%d@%s] epoch=%d | lr=%.4f | train_loss: %.3f (train_acc: %.3f) | val_loss: %.3f (val_acc: %.3f | %d/%d)', k, optimizer, i, lr, loss, train_accuracy(1,i), mean_val_loss, validation_accuracy(1,i), n_correct, n_total));
            end
    end
    
    disp(sprintf('[%d] checkpoint model saved to: %s', m, outFile));
    save(outFile, 'train_test_split', 'cvMethod', 'n_folds', 'n_train', 'n_test', 'seed', 'train_losses', 'validation_losses', 'train_accuracy', 'validation_accuracy', 'w_outs', 'w_inits', 'tau_m', 'tau_s', 'V_rest', 'V_thresh', 'T', 'dt', 'ts');

end

end

end