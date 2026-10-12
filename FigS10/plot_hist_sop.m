clear;
clc;
close all;

data = load('E:/p3_TCC/paper2/final/draft1/fig2/d/hist_qst/hist_cg_byNp_bins25_qst_L=0_bulk.txt');

phi = data(:,1);
P   = data(:,2);

mean_phi = 21.5;

figure('Color','w','Position',[100 100 700 550]);
hold on

% Left of mean (Hard)
idx_left = phi <= mean_phi+1;

area(phi(idx_left),P(idx_left),...
    'FaceColor','r',...
    'FaceAlpha',0.4,...
    'EdgeColor','none');

% Right of mean (Soft)
idx_right = phi >= mean_phi-1;

area(phi(idx_right),P(idx_right),...
    'FaceColor','b',...
    'FaceAlpha',0.4,...
    'EdgeColor','none');

% Original histogram curve
plot(phi,P,'-k','LineWidth',3.5);

% Mean line
xline(mean_phi,'--k','LineWidth',2.5);

xlabel('$\beta \Phi^i$','Interpreter','latex','FontSize',32);
ylabel('$P(\beta \Phi^i)$','Interpreter','latex','FontSize',32);

text(4,0.12,'Soft','FontSize',28);
text(30,0.12,'Hard','FontSize',28);

set(gca,...
    'FontSize',20,...
    'LineWidth',1.5,...
    'TickDir','out',...
    'Box','on');
axis square;
print('Phi distribution2','-dpng','-r300');