% % dx = load('stuck_dremoved_sio2_2.32_lav4_fps1_dx.txt');
% % dy = load('stuck_dremoved_sio2_2.32_lav4_fps1_dy.txt');
% 
% figure;
% 
% histogram(dx,200,'Normalization','pdf');
% hold on;
% histogram(dy,200,'Normalization','pdf');
% 
% box on
% axis square
% set(gcf,'units','centimeters','position',[0 0 25 18])
% set(gca,'fontsize',32);
% xlabel('$\Delta x,\Delta y$','FontSize',32,'interpreter','latex')
% ylabel('$p(\Delta x)$','FontSize',32,'interpreter','latex')
% legend('$\Delta x$','$\Delta y$','FontSize',28,'interpreter','latex');
% legend box off
% xlim([-0.07 0.07])
% exportgraphics(gcf,'uncertainity.jpg','Resolution',300)

% dx = load('stuck_dremoved_sio2_2.32_lav4_fps1_dx.txt');
% dy = load('stuck_dremoved_sio2_2.32_lav4_fps1_dy.txt');

figure;

[counts_dx, edges_dx] = histcounts(dx,300,'Normalization','probability');
centers_dx = (edges_dx(1:end-1) + edges_dx(2:end))/2;

% [counts_dy, edges_dy] = histcounts(dy,100,'Normalization','probability');
% centers_dy = (edges_dy(1:end-1) + edges_dy(2:end))/2;

% plot(centers_dx,counts_dx,'-o',...
%     'LineWidth',2.5,...
%     'MarkerSize',10,...
%     'MarkerIndices',1:1:length(centers_dx));

bar(centers_dx, counts_dx, ...
    'FaceColor', '[0.3 0.3 0.3]', ...
    'LineWidth', 2);

% hold on
% 
% plot(centers_dy,counts_dy,'-s',...
%     'LineWidth',2.5,...
%     'MarkerSize',10,...
%     'MarkerIndices',1:1:length(centers_dy));

box on
axis square

set(gcf,'units','centimeters','position',[0 0 25 18])
set(gca,'fontsize',24)

xlabel('$\Delta x (\mu m)$','FontSize',32,'Interpreter','latex')
ylabel('$p(\Delta x)$','FontSize',32,'Interpreter','latex')

xlim([-0.06 0.06])

exportgraphics(gcf,'uncertainity_pdf_bar.jpg','Resolution',300)