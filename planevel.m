

sx = linspace(-1, 1, 1920);
sy = linspace(-1, 1, 990);
[x,y] = meshgrid(sx,sy);
mx = 0;my = 0.8;z0 = 1;
z = mx*x + my*y + z0;

% surf(x,z,y)
% xlabel('x')
% ylabel('z')
% zlabel('y')

tilt = pi/3;
v = 5;
%T = [0;v*cos(tilt);v*sin(tilt)];
T = [0;10;50];
f = 10;

coeff = (1-mx*x-my*y)/(f*z0);
u1 = coeff.*([-f 0]*T(1:2) + x*T(3));
u2 = coeff.*([0 -f]*T(1:2) + y*T(3));

figure
quiver(x,y,u1,u2)
% selectx = 1:100:length(x);
% selectx = 1:100:length();
% quiver(x,y,u1,u2)