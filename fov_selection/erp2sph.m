function [phi theta] = erp2sph(m,n,W,H)
u = (m+0.5)/W;
v = (n+0.5)/H;
phi = (u-0.5)*2*pi;
theta = (0.5 - v)*pi;