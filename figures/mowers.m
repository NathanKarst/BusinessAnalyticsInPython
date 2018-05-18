
function mowers()

clc; clear; close all
x =[ 
   60.0000   18.4000    1.0000;
   85.5000   16.8000    1.0000;
   64.8000   21.6000    1.0000;
   61.5000   20.8000    1.0000;
   87.0000   23.6000    1.0000;
  110.1000   19.2000    1.0000;
  108.0000   17.6000    1.0000;
   82.8000   22.4000    1.0000;
   69.0000   20.0000    1.0000;
   93.0000   20.8000    1.0000;
   51.0000   22.0000    1.0000;
   81.0000   20.0000    1.0000;
   75.0000   19.6000         0;
   52.8000   20.8000         0;
   64.8000   17.2000         0;
   43.2000   20.4000         0;
   84.0000   17.6000         0;
   49.2000   17.6000         0;
   59.4000   16.0000         0;
   66.0000   18.4000         0;
   47.4000   16.4000         0;
   33.0000   18.8000         0;
   51.0000   14.0000         0;
   63.0000   14.8000         0];

b = [100,149,237]./255;
g = [60,179,113]./255;


idx = x(:,3) == 1;
plot(x(idx,1),x(idx,2),'s','Color','k','MarkerSize',10,'MarkerFaceColor',b); hold on
plot(x(~idx,1),x(~idx,2),'d','Color','k','MarkerSize',10,'MarkerFaceColor',g)
xlabel('Income ($1000)','FontSize',24)
ylabel('Yard Size (sq. ft.)','FontSize',24)
set(findall(gcf,'-property','FontSize'),'FontSize',24)
xmin = min(x(:,1))-5;
xmax = max(x(:,1))+5;
ymin = min(x(:,2))-1;
ymax = max(x(:,2))+1;
axis([xmin xmax ymin ymax  ])

plot([100 100],[ymin ymax],'k','LineWidth',2)

plot([xmin 100],[16 16],'k','LineWidth',2)

print -depsc cart_mowers_tree_map_example.eps

close all

idx = x(:,3) == 1;
plot(x(idx,1),x(idx,2),'s','Color','k','MarkerSize',10,'MarkerFaceColor',b); hold on
plot(x(~idx,1),x(~idx,2),'d','Color','k','MarkerSize',10,'MarkerFaceColor',g)
xlabel('Income ($1000)','FontSize',24)
ylabel('Yard Size (sq. ft.)','FontSize',24)
set(findall(gcf,'-property','FontSize'),'FontSize',24)
xmin = min(x(:,1))-5;
xmax = max(x(:,1))+5;
ymin = min(x(:,2))-1;
ymax = max(x(:,2))+1;
axis([xmin xmax ymin ymax  ])


print -depsc cart_mowers_tree_map_0.eps

plot([60 60],[ymin ymax],'k','LineWidth',2)

print -depsc cart_mowers_tree_map_1.eps

plot([xmin 60],[21 21],'k','LineWidth',2)

print -depsc cart_mowers_tree_map_2.eps

plot([60 xmax],[20 20],'k','LineWidth',2)

print -depsc cart_mowers_tree_map_3.eps

plot([85 85],[ymin 20],'k','LineWidth',2)

print -depsc cart_mowers_tree_map_4.eps

plot([62 62],[ymin 20],'k','LineWidth',2)

print -depsc cart_mowers_tree_map_5.eps

plot(70,16,'ko','MarkerSize',10)

print -depsc cart_mowers_tree_map_test.eps

