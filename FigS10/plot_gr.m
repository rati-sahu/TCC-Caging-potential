clear;
clc;
close all;

%% Load data

data = load('gr_bulk_soft_hard_avg.txt');

r       = data(:,1);
bulk_gr = data(:,2);
soft_gr = data(:,3);
hard_gr = data(:,4);

%% Plot

figure('Color','w','Position',[100 100 700 550]);

plot(r,bulk_gr,'k--','LineWidth',2.5);
hold on;

plot(r,soft_gr,'r-','LineWidth',2.5);

plot(r,hard_gr,'b-','LineWidth',2.5);

yline(1,'k:','LineWidth',1.5);

%% Labels

xlabel('$r/\sigma$','Interpreter','latex','FontSize',32);
ylabel('$g(r)$','Interpreter','latex','FontSize',32);

%% Legend

legend({'Bulk','Soft particles','Hard particles'}, ...
       'Location','northeast', ...
       'FontSize',16, ...
       'Box','off');

%% Axis formatting

ax = gca;

ax.FontSize = 28;
ax.LineWidth = 1.5;
ax.TickDir = 'out';
ax.Box = 'on';

xlim([0 2.9])

%% Optional: zoom near first few peaks
% xlim([0 5])

%% Save figure

% print('gr_soft_hard_bulk','-dpng','-r300');
% print('gr_soft_hard_bulk','-depsc','-painters');