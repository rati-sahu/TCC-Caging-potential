clear;
clc;
close all;

r = par_gr(:,end);
gr_crystal    = load('avg_gr_central_FCC_ids_crystal.txt');
gr_glass      = load('avg_gr_central_FCC_ids_glass_4-12-1.txt');

plot(r,gr_crystal,'LineWidth',2)
hold on
plot(r,gr_glass,'LineWidth',2)

xlabel('$r/\sigma$','FontSize',40,'interpreter','latex')
ylabel('$g(r)$','FontSize',40,'interpreter','latex')
legend('FCC in Crystal','FCC in Glass')
box on
axis square

set(gca,'FontSize',28)