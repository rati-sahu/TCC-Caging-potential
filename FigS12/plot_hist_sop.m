clear;
clc;
close all;

path = 'E:/p3_TCC/review/ref2/tcc_sop/';

files = {
    'softness_matrix_qst_dr=02_sigma02_0percent.txt'
    'softness_matrix_qst_dr=02_sigma02_1percent.txt'
    'softness_matrix_qst_dr=02_sigma02_3percent.txt'
    'softness_matrix_qst_dr=02_sigma02_5percent.txt'    
};

labels = {'experiment','1%','3%','5%'};

nbins = 30;
edges = linspace(0,60,nbins+1);
binCenters = (edges(1:end-1) + edges(2:end))/2;

styles = {'-o','-','--','-.'};
colors = lines(length(files));

figure;
hold on;

% ---------------- MAIN PLOT (histograms) ----------------
for i = 1:length(files)

    data = load([path files{i}]);
    xdata = data(:,1);

    counts = histcounts(xdata, edges, 'Normalization','probability');

    if i == 1
        plot(binCenters, counts, ...
            'LineStyle','-', ...
            'Marker','o', ...
            'LineWidth',2.5, ...
            'Color',colors(i,:), ...
            'MarkerSize',7);
    else
        plot(binCenters, counts, ...
            'LineStyle',styles{i}, ...
            'LineWidth',2.5, ...
            'Color',colors(i,:));
    end
end

xlabel('$\beta \Phi$','Interpreter','latex','FontSize',32);
ylabel('$P(\beta \Phi)$','Interpreter','latex','FontSize',32);

set(gcf,'units','centimeters','position',[0 0 25 18])
set(gca,'fontsize',32)
box on
legend(labels,'Location','northeast','FontSize',28);
legend box off
axis square

% ---------------- INSET: MEAN VALUES (SQUARE, TOP RIGHT) ----------------
ax2 = axes('Position',[0.5 0.62 0.3 0.20]); % square inset (top-right)

hold(ax2,'on');

means = zeros(1,length(files));

for i = 1:length(files)
    data = load([path files{i}]);
    xdata = data(:,1);
    means(i) = mean(xdata);
end

plot(ax2, 1:length(files), means, 'o-', ...
    'LineWidth',2.5, ...
    'MarkerSize',7);

set(ax2,'XTick',1:length(files))
set(ax2,'XTickLabel',labels)
xtickangle(ax2,30)

ylim(ax2,[0 40]);          
xlim(ax2,[0.5 4.5]);
axis square
ylabel(ax2,'Mean $\beta \Phi$','Interpreter','latex','FontSize',18);

set(ax2,'FontSize',16)
box(ax2,'on');