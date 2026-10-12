clear;
clc;
close all;

q6 = load('E:/p3_TCC/review/ref1/FCC_q6_central_4-12-1.txt');
q4 = load('E:/p3_TCC/review/ref1/FCC_q4_central_4-12-1.txt');
% q6 = load('E:/p3_TCC/review/ref1/FCC_q6_all_crystal.txt');
% q4 = load('E:/p3_TCC/review/ref1/FCC_q4_all_crystal.txt');

edges = 0:0.01:0.7;

figure;

histogram(q6,'BinEdges',edges,'Normalization','probability');
hold on
histogram(q4,'BinEdges',edges,'Normalization','probability');

xline(0.19,'k--','LineWidth',1.5);
xline(0.57,'r--','LineWidth',1.5);

legend('FCC in glass Q_6','FCC in glass Q_4','FontSize',18);
legend box off;
xlim([0 0.7]);

box on
axis square
ax=gca;
ax.LineWidth=1.5;
ax.PlotBoxAspectRatio=[1 1 1];

set(gca,'fontsize',32);

set(gcf,'units','centimeters','position',[0 0 25 18])

xlabel('$ Q_4, Q_6$','FontSize',32,'interpreter','latex')
ylabel('$P(Q_4, Q_6)$','FontSize',32,'interpreter','latex')

%exportgraphics(gcf,'fcc_q4q6_glass.jpg','Resolution',300);