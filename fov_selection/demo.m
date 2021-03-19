clear all;
clc;

img_dis_rgb=imread('375.png');
[phi theta]=select_points(img_dis_rgb);
spoint_radian = [phi' theta'];
img_dis_rgb=imresize(img_dis_rgb,[512 1024]);
im2fov(img_dis_rgb,spoint_radian,'375');