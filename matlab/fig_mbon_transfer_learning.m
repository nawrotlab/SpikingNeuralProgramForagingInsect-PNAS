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


filePattern = ['model_cache/predictions/', dataSetName, '.*'];
f = dir(filePattern);
files = {f.name};
files = natsort(files)';
fprintf('models: %d\n', length(files));
N_models = length(files);

% eval on those data-sets
% label, dataSet
dataSets = {
    {'CS+/CS-/1 bg. odor', ['PoisonPulse', sparsityLevel, '_0-3-15-10sec.mat'], '#1'},
    {'CS+/CS-/2 bg. odors (distinct)', ['PoisonPulse', sparsityLevel, '_0-3-5-15-10sec.mat'], '#2'},
    {'CS+/CS-/2 bg. odors (distinct & CS+ similar)', ['PoisonPulse', sparsityLevel, '_0-3-8-15-10sec.mat'], '#3'},
    {'CS+/CS-/3 bg. odors (2 distinct & CS+ similar)', ['PoisonPulse', sparsityLevel, '_0-3-5-8-15-10sec.mat'], '#4'}
};

results = cell(N_models, length(dataSets));
x_accu = [];
x_accu_std = [];
err_low = [];
err_high = [];
labels = cell(1,length(dataSets));
colors = colormap(lines(length(dataSets)));

for i=1:N_models
   for j=1:length(dataSets)
      fileName = sprintf('model_cache/predictions/%s/%s', files{i}, dataSets{j}{2});
      data = load(fileName);
      results{i,j} = data.accu;
      labels{j} = dataSets{j}{3};
   end
end


c = categorical(labels);
c = reordercats(c, labels);
for j=1:length(dataSets)
   x_accu = [x_accu mean([results{:,j}])];
   x_accu_std = [x_accu_std std([results{:,j}])];
   err_low = [err_low min([results{:,j}])];
   err_high = [err_high max([results{:,j}])];
   
    x = 1:length(dataSets);
    %b = bar(c(j),x_accu(j),'FaceColor',colors(j,:)); 
    b = bar(c(j), double(x_accu(j)), 'FaceColor', [.4 .4 .4]); 
    hold on;
end

%title('transfer learning: seq. task');
er = errorbar(x,x_accu,x_accu_std,[]);    
er.Color = [0 0 0];                            
er.LineStyle = 'none'; 
ylim([0 100]);
ylabel('avg. accuracy');
yticks([0 10 25 50 80 100]);
set(gca,'box','off');

set(gca,'Units','centimeters','Position',[1 1 2 3]);

fig = gcf;
fig.Units               = 'centimeters';
fig.Position(3)         = 3; %10.5;
fig.Position(4)         = 4;
set(fig.Children, ...
    'FontName',     'Arial', ...
    'FontSize',     8);
set(fig, 'DefaultFigureRenderer', 'painters');
fig.PaperPositionMode   = 'auto';
set(fig, 'PaperUnits', 'centimeters', 'Units', 'centimeters');
set(fig, 'PaperSize', fig.Position(3:4), 'Units', 'centimeters');
mkdir('../figures/', dataSetName);
print(gcf, '-dpdf', ['../figures/', dataSetName, '/fig_transfer_learning.pdf']);