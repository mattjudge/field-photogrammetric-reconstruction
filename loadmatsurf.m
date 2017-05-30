load('./output/9900_9920_tripletrain_heatmap_neg_gsigma0_modeplain.mat')
downsample = 4;
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
