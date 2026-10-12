clear;
clc;
close all;

Ntot = 62732;

cluster = {'10B','11C','11E','11F','12D','12E','13A','13B','FCC','HCP'};
x = 1:length(cluster);

% 0% uncertainty

mean0 = [
3.857370842560252089e-01
2.870512754339863970e-01
4.474312971033685971e-01
3.141420458584242015e-01
2.284649601231934779e-01
1.650112349397911959e-01
1.333837638265232550e-02
1.335038987619997265e-02
4.006646903085418071e-02
1.679777994821748596e-02
];

std0 = [
9.651119146293732007e-03
9.386052970299421302e-03
1.198435526278373089e-02
1.117153636963266229e-02
9.632020234941241627e-03
7.106054215778990699e-03
1.439174110918388719e-03
1.106897955958475923e-03
2.110597054299383213e-03
1.470234417764168045e-03
];

% 1% uncertainty

mean1 = [
24625.3
18483.2
28137.9
22367.6
14882.6
12414.9
981.8
1063.7
3118.4
1349.2
];

std1 = [
150.81486074728255
55.541375968231506
172.85732716768345
141.68525368262962
117.47264456979856
104.40035121269149
37.667551612023374
37.01065912529644
69.29999198332486
62.69112997687773
];

% 3% uncertainty

mean3 = [
21542.8
15674.8
24709.9
19206.6
12084.0
10009.5
655.3
895.5
2523.6
1217.9
];

std3 = [
133.38565640027917
152.98932424627978
204.23540339519982
254.59257211823322
194.55704676121204
186.91605364738234
21.442688471572048
72.59055647182275
63.592801828858306
95.03268443599345
];

% 5% uncertainty

mean5 = [
15089.0
10420.566666666668
17573.366666666665
13198.233333333334
7172.566666666667
5974.1
257.46666666666664
572.3333333333334
1422.3
836.9
];

std5 = [
186.4471821297869
186.15782761304771
267.557726073476
249.642321143165
188.63606327480463
205.23029117017083
48.8641560566274
71.98674846124797
84.58831336905051
90.92943455760361
];

% Convert to fractions

frac1 = mean1/Ntot;
frac3 = mean3/Ntot;
frac5 = mean5/Ntot;

err1 = std1/Ntot;
err3 = std3/Ntot;
err5 = std5/Ntot;

frac0 = mean0;
err0  = std0;
% Colors (same color family, increasing darkness)

c1 = [0.70 0.85 1.00];   % light blue (1%)
c3 = [0.30 0.60 0.90];   % medium blue (3%)
c5 = [0.00 0.25 0.60];   % dark blue (5%)

% Plot

figure('Color','w');
hold on

% Colors: same family, increasing darkness
c0 = [0.85 0.15 0.15];   % red
c1 = [0.75 0.88 1.00];   % 1%
c3 = [0.35 0.62 0.90];   % 3%
c5 = [0.00 0.25 0.60];   % 5%

errorbar(x,frac0,err0,...
    'd',...
    'Color',c0,...
    'MarkerFaceColor',c0,...
    'MarkerSize',12,...
    'LineWidth',1.5,...
    'CapSize',18,...
    'LineStyle','none');

errorbar(x,frac1,err1,...
    'o',...
    'Color',c1,...
    'MarkerFaceColor',c1,...
    'MarkerSize',12,...
    'LineWidth',1.5,...
    'CapSize',18,...
    'LineStyle','none');

errorbar(x,frac3,err3,...
    's',...
    'Color',c3,...
    'MarkerFaceColor',c3,...
    'MarkerSize',12,...
    'LineWidth',1.5,...
    'CapSize',18,...
    'LineStyle','none');

errorbar(x,frac5,err5,...
    '^',...
    'Color',c5,...
    'MarkerFaceColor',c5,...
    'MarkerSize',12,...
    'LineWidth',1.5,...
    'CapSize',18,...
    'LineStyle','none');

set(gca,'XTick',x)
set(gca,'XTickLabel',cluster)

xlabel('Cluster type','FontSize',28)
ylabel('$N_c/N$','Interpreter','latex','FontSize',28)

% legend({'1% tracking uncertainty',...
%         '3% tracking uncertainty',...
%         '5% tracking uncertainty'},...
%         'Location','northeast')

 legend({'experiment',...
        '1% tracking uncertainty',...
        '3% tracking uncertainty',...
        '5% tracking uncertainty'},...
        'Location','northeast')
    
set(gca,'FontSize',22)
xtickangle(45)

box on
xlim([0.5 length(cluster)+0.5])

set(gcf,'Units','centimeters',...
         'Position',[0 0 24 18])
% Optional: logarithmic y-axis if rare motifs are hard to see
% set(gca,'YScale','log')