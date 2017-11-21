load('./output/31302_31600_singletrain_heatmap_neg_gsigma0.mat')
downsample = 5;
X = X(1:downsample:end,1:downsample:end);
Y = Y(1:downsample:end,1:downsample:end);
Z = Z(1:downsample:end,1:downsample:end);

s = surf(X, Y, Z);
s.EdgeColor = 'none';
xlabel('x [m]')
ylabel('y [m]')
zlabel('z [m]')
axis equal
set(gca,'Xdir','reverse')
colormap hot

title('Point cloud generated from SFT 31302-31600')