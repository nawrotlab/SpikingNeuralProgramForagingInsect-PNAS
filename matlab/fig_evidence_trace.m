plumeModelName = 'PoisonPulse_0-3-5-15-10sec';
plumeModelName = 'GaussianLowSparsity_15-0-3-15_10sec';
odorId = 15;
col_cs_minus = [0 158 227];
col_cs_plus = [243 146 0];

predictions = load(sprintf('model_cache/predictions/msp_classicalLabCondLowSparsity-0-15.odor-%d.1-sp.5/%s.mat', odorId, plumeModelName));
data = load(sprintf('../data/%s.mat', plumeModelName));
rng(1365);
N_trials = 3;

trialIdx = randsample(data.data.trial_ids, N_trials) + 1;
%trialIdx = data.data.trial_ids(1:N_trials)+1;
bgOdorIds = [3 5 15];
colors = [[0 0 0]; col_cs_plus; [192 192 192]] / 255;
odorIdx = odorId + 1;
stim_times = data.data.stimulus_times(:,odorIdx);
T_trial = data.data.T_trial;
sp_times = predictions.sp_times';
T_center = T_trial / 2;
tau_avg = 1.5;
tau = tau_avg;
idx_cues_model = 1;
idx_cues_true = 2;
idx_cues_bg = 3;
dt = 1/1000;
t = 1:ceil(T_trial / dt);
t = t .* dt;

t_all = 1:ceil((T_trial * N_trials) / dt);
t_all = t_all .* dt;

% bin spike times
X = zeros(3,N_trials,length(t));
X_sp = {};
for k=1:N_trials
   % model prediction
   sp = sp_times{trialIdx(k)};
   for i=1:length(sp)
        idx = round(sp(i) / dt);
        X(idx_cues_model,k,idx) = 1;
   end 
   % true sensory cues
   stim = stim_times{trialIdx(k)};
   for i=1:length(stim)
        idx = round(stim(i) / dt);
        X(idx_cues_true,k,idx) = 1;
   end 
   % background / distractor cues
   bg_stim = data.data.stimulus_times{trialIdx(k),bgOdorIds+1};
   for i=1:length(bg_stim)
        idx = round(bg_stim(i) / dt);
        X(idx_cues_bg,k,idx) = 1;
   end
end

f = figure();
f.Renderer='Painters';
%set(gca,'ydir','reverse')
%set(gca,'vis','on');
%set(gca,'xtick',[],'ytick',[]);
%set(gca,'box','off');
%set(gca,'ycolor',[.7 .7 .7],'xcolor',[.7 .7 .7]);
%set(gca,'xlim', [0 T_trial]);
%set(gca,'ylim', [0 2.5]);

% plot trials
subplot(3,1,1);
hold on;
ax = gca;
ax.YAxis.TickLength = [0 0];
%ax.YAxis.LineWidth = 0.0;

offsets = [[0 0.5] ; [-0.5 0] ; [-0.5 0]];
hndl = cell(size(X,1), 1);
for k=1:N_trials
    plot([0 T_trial], [k k] + max(max(offsets)), 'Color', [0 0 0 0.2], 'LineWidth', 1);
    for i=1:size(X,2)
        for j=1:size(X,1)
            sp_pos = t(squeeze(X(j,k,:)) == 1);
            hndl{j,1} = plot([sp_pos; sp_pos], [(ones(size(sp_pos))*k) + offsets(j,1); (ones(size(sp_pos))*k) + offsets(j,2)], ...
                'Color', colors(j,:), 'linewidth', 2);
        end
    end
end
xlim([0 T_trial]);
xticks([]);
%xlabel('time [sec]');
ylim([max(max(offsets)) N_trials+1]);
yticks([1 N_trials]);
ylabel('casting iteration');
try
legend([hndl{1}(1) hndl{2}(1) hndl{3}(1)], {'model output', 'sensory cue', 'background cue'}, 'Location', 'northoutside', 'NumColumns', 3);
catch
legend([hndl{1}(1) hndl{2}(1) hndl{3}(1)], {'model output', 'sensory cue', 'background cue'}, 'Location', 'northoutside');  
end



% plot smoothed data
subplot(3,1,2);
hold on;
ax.YAxis.TickLength = [0 0];
for k=1:N_trials
    %plot([0 T_trial], [k k] + 1, 'Color', [0 0 0 0.2], 'LineWidth', 1);
    for i=1:size(X,2)
        for j =1:size(X,1)-1
            %% convolute
            kernel = gausswin(round(tau / dt));
            evidence = conv(squeeze(X(j,k,:)), kernel, 'same');
            lineStyle = '-';
            if mod(j,2) == 0
                lineStyle = '--'; % make true sensory rate dashed
            end
            
            plot(t, ((evidence / max(evidence))*0.8) + k,'color',[colors(j,:) .6],'linewidth', 1,'LineStyle', lineStyle);
        end
    end
end
xlim([0 T_trial]);
xticks([]);
%xlabel('time [sec]');
ylim([max(max(offsets)) N_trials+1]);
yticks([1 N_trials]);
ylabel('accum. evidence');

% plot averaged density
subplot(3,1,3);
hold on;
ax.YAxis.TickLength = [0 0];

%% convolute
kernel = gausswin(round(tau_avg / dt));
%kernel = kernel./sum(kernel);
avg_evidence = sum(squeeze(X(1,:,:)), 1);
evidence = conv(avg_evidence, kernel, 'same');            
plot(t, ((evidence / max(evidence))*0.9),'color',[colors(1,:) .6],'linewidth', 1,'LineStyle', '-');

avg_evidence = sum(squeeze(X(2,:,:)), 1);
evidence = conv(avg_evidence, kernel, 'same');            
plot(t, ((evidence / max(evidence))*0.9),'color',[colors(2,:) .6],'linewidth', 1,'LineStyle', '--');

plot([T_center T_center], [0 1], 'LineWidth', 1, 'LineStyle', '--', 'Color', [1 0 1 0.8]);
legend({'model', 'true evidence', 'plume center'});

xlim([0 T_trial]);
xticks(0:2:T_trial);
yticks([ 0 1]);
xlabel('time [sec]');
%ylim([max(max(offsets)) N_trials+1]);
ylabel('avg. evidence');

%lbwh = get(gca, 'position');
%lbwh(end) = lbwh(end)*0.5;
%set(gca, 'position', lbwh);
print(sprintf('../figures/fig_evidence_trace_%s.pdf', plumeModelName),'-dpdf','-fillpage');
