%This code calculated the contour of softness and the cluster lfs on top of the contour map.
coolwarm = [...
    0.2298, 0.2980, 0.7549;  % cool blue
    1, 1, 1;  % white
    1, 0, 0   % warm red
];

n = 256; 
x = linspace(0, 1, size(coolwarm, 1));
coolwarm_map = interp1(x, coolwarm, linspace(0, 1, n), 'linear');

colormap(coolwarm_map);
grey = [0.2, 0.2, 0.2];

m = 6;
t = 8;

txt_folder_path = ['/media/rati/rati/p3_TCC/tcc/xyzLinked/qst/feat/TS_4-12-1/'];
txt_file_path = [txt_folder_path, 'QTS_feat_TS_4-12-1_featum_y0-55_t=', num2str(t), '.txt'];
txt_frame = load(txt_file_path);
clu = '10b';
path_id = ['/media/rati/rati/p3_TCC/tcc/codes_new/further_analysis/data/ids/',clu,'/qst4/'];
clust_10b = load(strcat(path_id,'xyz_10b_TS_4-12-1_t=', num2str(t), '.txt'));
clu = '11c';
path_id = ['/media/rati/rati/p3_TCC/tcc/codes_new/further_analysis/data/ids/',clu,'/qst4/'];
clust_11f = load(strcat(path_id,'xyz_',clu,'_TS_4-12-1_t=', num2str(t), '.txt'));
min_x = min(txt_frame(:, 1));
max_x = max(txt_frame(:, 1));
min_y = min(txt_frame(:, 2));
max_y = max(txt_frame(:, 2));
min_z = min(txt_frame(:, 3));
max_z = max(txt_frame(:, 3));

cutoff = 3;
x_bounds = [50,90]; %[min_x + cutoff, max_x - cutoff];
y_bounds = [5,45];%[min_y + cutoff, max_y - cutoff];
z_bounds = [1,50];%[min_z + cutoff, max_z - cutoff-1];

within_bounds = ...
    (txt_frame(:, 1) >= x_bounds(1)) & (txt_frame(:, 1) <= x_bounds(2)) & ...
    (txt_frame(:, 2) >= y_bounds(1)) & (txt_frame(:, 2) <= y_bounds(2)) & ...
    (txt_frame(:, 3) >= z_bounds(1)) & (txt_frame(:, 3) <= z_bounds(2));
clust_id_10b = (clust_10b(:, 1) >= x_bounds(1)) & (clust_10b(:, 1) <= x_bounds(2)) & ...
    (clust_10b(:, 2) >= y_bounds(1)) & (clust_10b(:, 2) <= y_bounds(2)) & ...
    (clust_10b(:, 3) >= z_bounds(1)) & (clust_10b(:, 3) <= z_bounds(2));
clust_id_11f = (clust_11f(:, 1) >= x_bounds(1)) & (clust_11f(:, 1) <= x_bounds(2)) & ...
    (clust_11f(:, 2) >= y_bounds(1)) & (clust_11f(:, 2) <= y_bounds(2)) & ...
    (clust_11f(:, 3) >= z_bounds(1)) & (clust_11f(:, 3) <= z_bounds(2));
txt_frame_in = txt_frame(within_bounds, :);
clust_in_10b = clust_10b(clust_id_10b,:);
clust_in_11f = clust_11f(clust_id_11f,:);
% Path for id and sop data
path_sop = ['/media/rati/rati/p3_TCC/tcc/softness/TS_4-12-1/'];
sop = load([path_sop, 'sop_Avgsop_TS_4-12_1_rmin=0.75_rmax=1.3_L=0-3_t=', num2str(t), '.txt']);
sop_in = sop(within_bounds, :);
%clust = txt_frame(idt, :);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  data loded   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
savepath = '/media/rati/rati/p3_TCC/paper2/final/draft1/si/fig3/cg_11c_qst_t=8/';
xyzS = [txt_frame_in(:,1:3) sop_in(:,2)];

for i=1:1:40
    
    %colormap(coolwarm_map);
    colormap("parula");
    %caxis([0.03, 0.065]);  %sop
    y_min = i;
    y_max = i+3.2;
    ysec = find((xyzS(:,3)>y_min)&(xyzS(:,3)<y_max));
    pos_sec = xyzS(ysec,:);
    id_clust_10b = find((clust_in_10b(:,3)>y_min)&(clust_in_10b(:,3)<y_max));
    clust_sec_10b = clust_in_10b(id_clust_10b,:);

    id_clust_11f = find((clust_in_11f(:,3)>y_min)&(clust_in_11f(:,3)<y_max));
    clust_sec_11f = clust_in_11f(id_clust_11f,:);

    zl= min(xyzS(:,2)):1:max(xyzS(:,2));
    xl= min(xyzS(:,1)):1:max(xyzS(:,1));
    %zl= 4:1:46;
    %xl= 49:1:91;

    [xg,zg]=meshgrid(xl,zl);
    numLevels = 15;
    s2_sec=griddata(pos_sec(:,1),pos_sec(:,2),pos_sec(:,4),xg,zg);
    for j=1:1
        try
            contourf(xg,zg,s2_sec,numLevels, 'LineColor', 'k', 'LineWidth', 0.2)
        catch
        end
    end

    colorbar;
    hold on

    
    %%scatter(clust_sec_10b(:,1),clust_sec_10b(:,2),40,"white","o","filled")
    scatter(clust_sec_11f(:,1),clust_sec_11f(:,2),30,"white","o","filled")
    min(clust_sec_11f)
    max(clust_sec_11f)

    ax=gca;
    set(gca,'fontsize',32);
    xlabel('$x$','FontSize',40,'interpreter','latex')
    ylabel('$y$','FontSize',40,'interpreter','latex')
    ax.PlotBoxAspectRatio=[1 1 1];
    set(gcf,'units','centimeters','position',[0 0 25 18])
    dispstr = strcat('d)');
    annotation('textbox', [0.1, 0.95, 0, 0], 'String', dispstr,'FontSize',40,'Color','black')
    ax.LineWidth = 1.5;
    box on;
    axis tight;
    xticks([]); yticks([]);
    %title(sprintf('y = %d', i), 'FontWeight', 'normal');
    hold off;
    fname = [savepath,'fig3a_xy_L=1_11c_z=',num2str(i),'.jpg'];
    exportgraphics(gcf,fname,'Resolution',300)
    figure;
end