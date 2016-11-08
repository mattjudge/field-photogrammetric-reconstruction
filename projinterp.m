% gxs, gys = np.zeros_like(vxs), np.zeros_like(vys)
%     for posx in range(f1.shape[1]):
%         for posy in range(f1.shape[0]):
%             gxs[posy, posx] = -2e-06 * (posx - cenx) + 0.0002
%             gys[posy, posx] = -9e-06*posy - 0.0006

im1 = imread('vlcsnap-2016-10-20-13h36m38s764cropped.png');


cenx = 1166;
ceny = -597;
gxs = repmat(-2e-06 * ((1:1:size(im1,2)) - cenx) + 0.0002, size(im1,1), 1);
gys = repmat(-9e-06 * ((1:1:size(im1,1))' - cenx) - 0.0006, 1, size(im1,2));
size(gxs)
size(gys)
sz = size(im1);
[y, x] = ndgrid( 1:sz(1), 1:sz(2) );
im2 = interp2( x, y, im1(:,:,1), x + gxs, y + gys );
imshow(im2);
%im2 = interp2(im1(:,:,1), gxs, gys);