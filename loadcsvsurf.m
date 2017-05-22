M = csvread("14550_14800_singletrain_heatmap.csv");
x = M(:,1);
y = M(:,2);
z = M(:,3);
sh = [size(unique(M(:,1)), 1), size(unique(M(:,2)), 1)]
X = reshape(M(:,1), sh);
Y = reshape(M(:,2), sh);
Z = reshape(M(:,3), sh);

% to crop by x
% mask = X(:,1) > -20 & X(:,1) < 20;
% X = X(mask,:);
% Y = Y(mask,:);
% Z = Z(mask,:);

s = surf(X, Y, Z);
s.EdgeColor = 'none';
xlabel('x')
ylabel('y')
zlabel('z')

