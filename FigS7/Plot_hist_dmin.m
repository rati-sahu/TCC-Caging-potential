path = 'E:/p3_TCC/paper2/final/draft1/fig5/a/dmin_hist/';
pathc = 'E:/p3_TCC/paper2/final/draft1/fig5/a/dmin_cen/';
dia=1.6;
histi = load(strcat(path,'hist_10b_rich_d6_rmax=1.15_ra=1.5_t0-24_aveps.txt'));
histo = load(strcat(path,'hist_10b_poor_d6_rmax=1.15_ra=1.5_t0-24_aveps.txt'));

bins = 0:0.005:0.4;  
bins=bins';


bar(histo(:,1), histo(:,2), 'FaceColor', 'red','FaceAlpha',0.5, 'EdgeColor', 'none', 'BarWidth', 1)
hold on
bar(histi(:,1), histi(:,2), 'FaceColor', 'blue','FaceAlpha',0.65, 'EdgeColor', 'none', 'BarWidth', 1)

xlabel('$D_{min}$', 'Interpreter', 'latex')
ylabel('$P(D_{min}$)', 'Interpreter', 'latex')
legend('show')

set(gca, 'YScale', 'log')


hold off
axis square;
set(gca,'fontsize',32);
xlabel('$D^2_{min}$','FontSize',35,'interpreter','latex')
ylabel('$P(D^2_{min})$','FontSize',35,'interpreter','latex')
lgd = legend('$10B_{poor}$','$10B_{rich}$','interpreter','latex','Location','northeast');
set(lgd,'FontSize',26);
legend boxoff
ax=gca;
ax.LineWidth = 1.5;
ax.PlotBoxAspectRatio=[1 1 1];
set(gcf,'units','centimeters','position',[0 0 25 18])
dispstr = strcat('a)');
annotation('textbox', [0.08, 0.95, 0, 0], 'String', dispstr,'FontSize',40,'Color','black')
box on;
axis tight;
%ylim([0,4])
%yticks([0 0.4 0.8 1.2 1.5])
xlim([0,0.35])
%xticks([0 20 40 60 80 100])
%exportgraphics(gcf,'fig5a_10b_new.jpg','Resolution',300)