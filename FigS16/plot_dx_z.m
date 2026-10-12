clear;
clc;
close all;

x = load('E:/p3_TCC/tcc/xyzLinked/dec/bulk/d6/STS_ASM385_70_2-12_6_featumd_y0-55_t=0-24_PosX.dat');
y = load('E:/p3_TCC/tcc/xyzLinked/dec/bulk/d6/STS_ASM385_70_2-12_6_featumd_y0-55_t=0-24_PosY.dat');
z = load('E:/p3_TCC/tcc/xyzLinked/dec/bulk/d6/STS_ASM385_70_2-12_6_featumd_y0-55_t=0-24_PosZ.dat');

f1 = 1;
f2 = 10;

xyz0 = [x(:,f1), y(:,f1), z(:,f1)];
dx = x(:,f2) - x(:,f1);
z0 = z(:,f1);

figure;

scatter(dx,z0,20,'.')
ylim([5,70])
ylabel('$z (\mu m)$','FontSize',40,'interpreter','latex')
xlabel('$\Delta x (\mu m)$','FontSize',40,'interpreter','latex')
set(gcf,'units','centimeters','position',[0 0 25 18])
set(gca,'fontsize',32)
box on
axis square