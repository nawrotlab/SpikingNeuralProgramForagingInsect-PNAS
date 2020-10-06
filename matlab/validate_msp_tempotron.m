function [mean_loss, validation_errors, predictions, spiketimes] = validate_msp_tempotron(ts, trials, labels, w, V_thresh, V_rest, tau_m, tau_s)
    
    memo_exp = memoize(@exp);
    memo_exp.CacheSize = size(trials, 2)*10;
    validation_errors = zeros(1, size(trials, 1));
    predictions = zeros(1, size(trials, 1));
    dataFormatType = iscell(trials{1});
    %dataFormatType = size(trials{1},2) ~= size(trials{2},2) || size(trials{1},2) + size(trials{2},2) == 0;
    spiketimes = cell(1, size(trials, 1));
    for j=1:size(trials, 1)
       if dataFormatType == 0
            pattern = cell(trials(j,:));
        else
            pattern = trials{j};
       end
        
       [v_t, t_sp, ~, ~, ~] = MSPTempotron(memo_exp, ts, pattern, w, V_thresh, V_rest, tau_m, tau_s); 
       validation_errors(1, j) = abs(length(t_sp) - labels(j));
       predictions(1,j) = length(t_sp);
       spiketimes{1,j} = t_sp;
    end
    
    mean_loss = mean(validation_errors);
end