paths2 = 'E:/p3_TCC/review/ref3/2b/';
%bulk = load(strcat(paths2,'hist_cg_byNp_bins25_qst_L=1_bulk.txt'));
hist_1 = load(strcat(paths2,'hist_cg_central_byNp_0-25_d6_L=0_10b_sigma=0.01.txt'));
hist_5 = load(strcat(paths2,'hist_cg_central_byNp_100_d6_L=0_10b_sigma=0.01.txt'));

%plot(bulk(:,1), bulk(:,2), '-*k', 'linewidth', 2.5, 'markersize', 10) % Purple (#CC79A7)
hold on;
plot(hist_1(:,1), hist_1(:,2), 'o-', 'Color', 'red', 'linewidth', 2.5, 'markersize', 10) % Purple (#CC79A7)
plot(hist_5(:,1), hist_5(:,2), 's-', 'Color', 'blue', 'linewidth', 2.5, 'markersize', 10)   % Blue-Green (#009E73)  ,'color',[0.2,0.6,0.2]


axis square;
set(gca,'fontsize',28);
xlabel('$\beta \Phi^i$','FontSize',40,'interpreter','latex')
ylabel('$P(\beta \Phi^i)$','FontSize',40,'interpreter','latex')
lgd = legend('$10B\ poor$','$10B\ rich$','interpreter','latex','Location','northeast');
%lgd = legend('$0-25\%$','$25-50\%$','$50-75\%$','$75-100\%$','$100\%$','interpreter','latex','Location','northwest');
set(lgd,'FontSize',24);
legend boxoff
ax=gca;
ax.LineWidth = 1.5;
ax.PlotBoxAspectRatio=[1 1 1];
set(gcf,'units','centimeters','position',[0 0 25 18])
dispstr = strcat('a)');
title('$\delta=0.01$','FontSize',32,'interpreter','latex')
annotation('textbox', [0.1, 0.95, 0, 0], 'String', dispstr,'FontSize',40,'Color','black')
box on;
axis tight;
%xlim([0,44])
%xticks([-4 -3.5 -3 -2.5 -2])
%xlim([16,33])
%xticks([-4 -3.5 -3 -2.5 -2])

hold off;
exportgraphics(gcf,'fig5b_sigma=0.01.jpg','Resolution',300)