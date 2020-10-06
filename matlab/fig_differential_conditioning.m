close all;
clear all;
odorId = 15;

if exist('sparsityLevel', 'var') == 0
    sparsityLevel = 'ConnectivityMediumSparsity';
end

if exist('dataSetName', 'var') == 0
   %dataSetName = 'msp_classicalLabCondConnectivityLowSparsityAPL-0-15.odor-15.1-sp'; 
   dataSetName = sprintf('msp_classicalLabCond%s-0-15.odor-%d.1-sp', sparsityLevel, odorId); 
end

filePattern = ['model_cache/', dataSetName, '.*.mat'];
f = dir(filePattern);
files = {f.name};
files = natsort(files)';
fprintf('models: %d\n', length(files));
col_cs_minus = [0 158 227] / 255;
col_cs_plus = [243 146 0] / 255;
X_train = [];
X_test = [];
X_behavior_resp = {}; % col 1&2: cs+/cs- response on TrainSet, col 3&4: cs+/cs- on TestSet
N_models = length(files);

batch_size = 4; % batch size used to compute accuarcy score over
eval_behavior_at_samples = 20;

eval_beh_at_epoch = floor(40 / batch_size);


for j=1:N_models
   data = load(sprintf('model_cache/%s', files{j}));
   X_train = [X_train data.train_accuracy(:,1)];
   X_test = [X_test data.validation_accuracy(:,1)];
   
   if (length(find(data.train_accuracy == 0)) > 3)
       warning(sprintf('outlier (prob. perfectly converged model): %s', files{j}));
      continue; 
   end
   
   % compute behavior response:
   % correct CS+ response: 1 or more spikes
   % correct CS- response: exactly 0 spikes
   beh_resp = cell(length(data.predictions),6); 
   for i=1:length(data.predictions)
       % compute TrainSet behavior response to CS++ and CS-
       idx_cs =  find(data.predictions{i,2} == 1);
       idx_us =  find(data.predictions{i,2} == 0);
       idx_cs_behav = find(data.predictions{i,1} >= data.predictions{i,2});
       idx_us_behav = find(data.predictions{i,1} == data.predictions{i,2});
       res_vec = zeros(1, length(data.predictions{i,2}));
       res_vec(intersect(idx_cs_behav, idx_cs)) = 1;
       res_vec(intersect(idx_us_behav, idx_us)) = 1;
       beh_resp{i,1} = res_vec; % correctness vector over all train samples
       beh_resp{i,2} = data.predictions{i,1};
       beh_resp{i,3} = data.predictions{i,2};
       
       % compute TestSet behavior response to CS++ and CS-
       idx_cs =  find(data.predictions{i,4} == 1);
       idx_us =  find(data.predictions{i,4} == 0);
       idx_cs_behav = find(data.predictions{i,3} >= data.predictions{i,4});
       idx_us_behav = find(data.predictions{i,3} == data.predictions{i,4});
       res_vec = zeros(1, length(data.predictions{i,4}));
       res_vec(intersect(idx_cs_behav, idx_cs)) = 1;
       res_vec(intersect(idx_us_behav, idx_us)) = 1;
       beh_resp{i,4} = res_vec; % correctness vector over all test samples
       beh_resp{i,5} = data.predictions{i,3}; % predictions (TEST)
       beh_resp{i,6} = data.predictions{i,4}; % ground truth (TEST)
   end
   X_behavior_resp{end+1} = beh_resp;
end

N_models = length(X_behavior_resp);
xs = 1:size(X_train,1);
xs = xs * batch_size;

fig = figure();
% plut learning & test accuracy as function of training samples present
%subplot(2,1,1);
hold on;
sem = std(X_train, [], 2) / sqrt(size(X_train,1));
mu_plus_sem = mean(X_train, 2) + sem;
mu_minus_sem = mean(X_train, 2) - sem;
plot(0, 50, 'sk', 'MarkerFaceColor', [0 0 0]);
h1 = plot(xs, mean(X_train, 2), 'k', 'LineWidth', 1.5);
h2 = plot(xs, mu_plus_sem, 'Color', [.3 .3 .3]);
h3 = plot(xs, mu_minus_sem, 'Color', [.3 .3 .3]);

%fill([xs; fliplr(xs)]', [mu_minus_sem, fliplr(mu_plus_sem)], 'g');    % fill area defined by x & yy in blue

%fill(xs, [mean(X_train, 2) - sem, mean(X_train, 2) + sem], 0, ...
%    'FaceColor', [.3 .3 .3]);
%plot(xs, mean(X_train, 2) - sem, 'Color', [.3 .3 .3]);

%sem = std(X_test, [], 1);
%plot(xs, mean(X_test, 2), 'b');
%plot([eval_behavior_at_samples eval_behavior_at_samples], [0 100], '-.k');
%legend({'train', 'test', 'behavior response evaluation'},'Location','SouthEast');
%legend({'train', 'test'},'Location','SouthEast');
leg = legend([h1, h2],{'mean', 's.e.m.'}, 'Location', 'best');
leg.ItemTokenSize = [10,5];

%title(sprintf('training (N=%d models)', size(X_train,2)));
%plot(xs, median(X_train, 2) + sem, 'k');
%plot(xs, median(X_train, 2) - sem, 'k');
xlabel('# trials');
ylabel('accuracy [%]');
ylim([0 100]);
xlim([0 100]);
xticks([0 20 50 100]);
xticklabels(floor([0 20 50 100] / 2));
ax1 = gca();

% compute behav. score
B_test = zeros(2, length(X_behavior_resp{1, end}{eval_beh_at_epoch, 4}));
B_train = zeros(2, length(X_behavior_resp{1, end}{eval_beh_at_epoch, 2}));
B_norm_train = zeros(2, length(X_behavior_resp{1, end}{eval_beh_at_epoch, 2}));
B_norm_test = zeros(2, length(X_behavior_resp{1, end}{eval_beh_at_epoch, 4}));
fprintf("samples in epoch %d: %d (train) | %d (test)\n", eval_beh_at_epoch, size(B_train,2), size(B_test, 2));
fprintf("eval behavior at epoch: %d (%d samples)\n", eval_beh_at_epoch, eval_beh_at_epoch * batch_size);
for i=1:length(X_behavior_resp)
    % compute behavior learning curve on train set
    v_beh = X_behavior_resp{1, i}{eval_beh_at_epoch, 1};
    v_true = X_behavior_resp{1, i}{eval_beh_at_epoch, 3};
    for j=1:length(v_true)
        if (v_true(j) == 1 && v_beh(j) == 1)
            B_train(1,j) = B_train(1,j) + 1;
        end
        
        if (v_true(j) == 1)
           B_norm_train(1,j) = B_norm_train(1,j) + 1;
        end
        
        if (v_true(j) == 0 && v_beh(j) == 1)
            B_train(2,j) = B_train(2,j) + 1;
        end
        
        if (v_true(j) == 0)
           B_norm_train(2,j) = B_norm_train(2,j) + 1;
        end
    end
    % compute behavior learning curve on test set
    v_beh = X_behavior_resp{1, i}{eval_beh_at_epoch, 4};
    v_true = X_behavior_resp{1, i}{eval_beh_at_epoch, 6};
    for j=1:length(v_true)
        if (v_true(j) == 1 && v_beh(j) == 1)
            B_test(1,j) = B_test(1,j) + 1;
        end
        
        if (v_true(j) == 1)
           B_norm_test(1,j) = B_norm_test(1,j) + 1;
        end
        
        if (v_true(j) == 0 && v_beh(j) == 1)
            B_test(2,j) = B_test(2,j) + 1;
        end
        
        if (v_true(j) == 0)
           B_norm_test(2,j) = B_norm_test(2,j) + 1;
        end
    end
end
%B_train = B_train / length(X_behavior_resp);
%B_test = B_test / length(X_behavior_resp);
B_train = B_train ./ B_norm_train;
B_test = B_test ./ B_norm_test;


fig2 = figure();
%subplot(2,1,2);
ax = gca;
ax2 = ax;
hold on;
%title('behavior response (train)');
plot(1, 0, '-s', 'MarkerFaceColor', col_cs_plus, 'MarkerEdgeColor', col_cs_plus);
plot(1, 100, '-s', 'MarkerFaceColor', col_cs_minus, 'MarkerEdgeColor', col_cs_minus);

h1 = plot(2:size(B_train,2), B_train(1,2:end) * 100, '-o', ...
    'Color', col_cs_plus, 'MarkerFaceColor', col_cs_plus, ...
    'MarkerEdgeColor', [1 1 1]);
h2 = plot(2:size(B_train,2), B_train(2,2:end) * 100, '-d', ...
    'Color', col_cs_minus, 'MarkerFaceColor', col_cs_minus, ...
    'MarkerEdgeColor', [1 1 1]);

legend([h1,h2], {'CS+', 'CS-'},'Location','SouthEast');
xlabel('trial');
ylabel('correct responders [%]');
ylim([0 100]);
xlim([0, eval_behavior_at_samples]);
xticks([1 5 10 15 20]);
ax.XRuler.MinorTick = 'on';
ax.XRuler.MinorTickValues = 1:1:20;
ax.XRuler.MinorTickValuesMode = 'manual';

%axp = get(gca,'position');
%axp(1) = 1.1 * axp(1);
%set(gca, 'Position', axp);

%yh = get(gca,'ylabel'); % handle to the label object
%p = get(yh,'position'); % get the current position property
%p(1) = 0.9*p(1) ;        % double the distance, 
                       % negative values put the label below the axis
%set(yh,'position',p);   % set the new position

%subplot(1,3,3);
%hold on;
%title('behavior response (test)');
%plot(1:size(B_test,2), B_test(1,:) * 100, '-or');
%plot(1:size(B_test,2), B_test(2,:) * 100, '-db');
%legend({'CS+', 'CS-'},'Location','West');
%xlabel('trial');
%ylabel('% correct');
%ylim([0 100]);


%single-col figure: 85 mm

% model fitting learning curve
set(ax1,'Units','centimeters','Position',[1 1 3 9]);
fig.Units               = 'centimeters';
fig.Position(3)         = 4.5; % 4.5 ; %10.5;
fig.Position(4)         = 10.5;
set(fig.Children, ...
    'FontName',     'Arial', ...
    'FontSize',     8);
set(fig, 'DefaultFigureRenderer', 'painters');
fig.PaperPositionMode   = 'auto';
set(fig, 'PaperUnits', 'centimeters', 'Units', 'centimeters');
set(fig, 'PaperSize', fig.Position(3:4), 'Units', 'centimeters');
mkdir('../figures/', dataSetName);
print(fig, '-dpdf', ['../figures/', dataSetName, '/fig_learning.pdf']);

% behavioral learning curve
set(ax2,'Units','centimeters','Position',[1 1 3 5]);
fig2.Units               = 'centimeters';
fig2.Position(3)         = 4.5; % 4.5 ; %10.5;
fig2.Position(4)         = 6.5;
set(fig2.Children, ...
    'FontName',     'Arial', ...
    'FontSize',     8);
set(fig2, 'DefaultFigureRenderer', 'painters');
fig2.PaperPositionMode   = 'auto';
set(fig2, 'PaperUnits', 'centimeters', 'Units', 'centimeters');
set(fig2, 'PaperSize', fig.Position(3:4), 'Units', 'centimeters');
mkdir('../figures/', dataSetName);
print(fig2, '-dpdf', ['../figures/', dataSetName, '/fig_differential_conditioning.pdf']);
