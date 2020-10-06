function [rmse, accu,y,pred_y,sp_times] = rmse_mbon_task(modelName, dataSetFileName, target_odor_id, reward_size, w_idx)
    
    load(modelName);
    [filepath,name,ext] = fileparts(modelName);
    
    %reward_size = 1; % no of spikes for individual pattern
    %target_idx = 2; % idx of rewarded odor source (data.rewards contains multiple counts for each odor)


    load(dataSetFileName);
    N_samples = size(data.trials, 1);
    N_syn = size(w_outs, 2);
    samples = data.trials;
    rewards = double(data.targets);
    rewards(:, target_odor_id+1) = rewards(:, target_odor_id+1) .* reward_size;
    y = rewards(:,target_odor_id+1)';
    % data-set might have different duration than model
    T = double(data.T_trial);
    ts = 0:dt:T;
    if (w_idx <= 0)
        w_out = w_outs(end,:);
    else
        disp(sprintf('uusing weights of epoch %d', w_idx));
        w_out = w_outs(w_idx,:);
    end

    [~,dataSetName,~] = fileparts(dataSetFileName);
    disp(sprintf('computing RMSE over %d samples from file: %s', N_samples, dataSetFileName));
        
    [mean_val_loss, ~, pred_y, sp_times] = validate_msp_tempotron(ts, samples, rewards, w_out, V_thresh, V_rest, tau_m, tau_s);

    accu = (length(find(y == pred_y)) * 100)/length(y);
    rmse = sqrt(mean((y-pred_y).^2));
    disp(sprintf('RMSE=%.2f | accuracy=%.2f | N_samples=%d | N_syn=%d', rmse, accu, N_samples, N_syn));
    
    % only save if datSet contains > 1 sample
    if N_samples > 1
        targetPath = [filepath filesep 'predictions' filesep name];
        [status, msg, msgID] = mkdir(targetPath);
        outFile = fullfile(targetPath, strcat(dataSetName,ext));
        save(outFile, 'pred_y', 'y', 'accu', 'rmse', 'sp_times');
        disp(sprintf('saved results to: %s', outFile));
    else
        if ~isfield(data, 'predictions')
            [data(:).predictions] = {};
        end
        
        exists = 0;
        for i=1:length(data.predictions)
            if 1 == strcmp(data.predictions{i}{1}, name)
               data.predictions{i} = {name, pred_y,y,accu,rmse,sp_times}; 
               exists = 1;
            end
        end
        
        if (exists < 1)
            data.predictions{end+1} = {name, pred_y,y,accu,rmse,sp_times}; 
        end
        save(dataSetFileName, 'data');
        disp(sprintf('saved results to dataSet: %s', dataSetFileName));
    end
end