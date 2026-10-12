colors = [  0.00, 0.2, 0.9;  % Blue (#0072B2)
            0.9, 0.5, 0.00;  % Orange (#E69F00)
            0.00, 0.7, 0.5;  % Green (#009E73)
            1, 0.2, 0.5;  % Purple (#CC79A7)
            0.5, 0.89, 0.26;  % Yellow (#F0E442)
            0.00, 0.00, 0.00]; % Black (#000000)


paths2 = 'E:/p3_TCC/review/ref3/2/sigma=0.03/';
bulk = load(strcat(paths2,'hist_cg_byNp_bins25_qst_L=0_bulk_sigma=0.03.txt'));
hist_10b = load(strcat(paths2,'hist_cg_byNp_bins25_qst_L=0_center_10B_sigma=0.03.txt'));
hist_11f = load(strcat(paths2,'hist_cg_byNp_bins25_qst_L=0_center_11F_sigma=0.03.txt'));
hist_13a = load(strcat(paths2,'hist_cg_byNp_bins25_qst_L=0_center_13A_sigma=0.03.txt'));
hist_fcc = load(strcat(paths2,'hist_cg_byNp_bins25_qst_L=0_center_FCC_sigma=0.03.txt'));

plot(bulk(:,1), bulk(:,2), '-k', 'linewidth', 3.5, 'markersize', 10) % Purple (#CC79A7)
hold on;
plot(hist_10b(:,1), hist_10b(:,2), '--', 'linewidth', 2.5, 'markersize', 10) % Purple (#CC79A7)
plot(hist_11f(:,1), hist_11f(:,2), '-', 'linewidth', 2.5, 'markersize', 10)   % Red (#D55E00)
plot(hist_fcc(:,1), hist_fcc(:,2), '-*', 'linewidth', 2.5, 'markersize', 10)   % Blue-Green (#009E73)
plot(hist_13a(:,1), hist_13a(:,2), '-.', 'linewidth', 2.5, 'markersize',10)   % Orange (#E69F00)
title('$\delta=0.03$','FontSize',5,'interpreter','latex')
ax=gca;
ax.LineWidth = 1.5;
ax.PlotBoxAspectRatio=[1.7 1 1];
set(gcf,'units','centimeters','position',[0 0 25 18])
%axis square;
set(gca,'fontsize',32);
xlabel('$\beta \Phi^i$','FontSize',40,'interpreter','latex')
ylabel('$P(\beta \Phi^i)$','FontSize',40,'interpreter','latex')
lgd = legend('$bulk$','$10B$','$11F$','$FCC$','$13A$','interpreter','latex','Location','northeast');
set(lgd,'FontSize',24);
legend boxoff

dispstr = strcat('b)');
annotation('textbox', [0, 0.85, 0, 0], 'String', dispstr,'FontSize',40,'Color','black')
box on;
axis tight;
%xlim([11,40])
xticks([0 10 20 30 40 50 60 70])
%xlim([16,33])
%xticks([-4 -3.5 -3 -2.5 -2])

hold off;
exportgraphics(gcf,'fig2a_sigma=0.03.jpg','Resolution',300)