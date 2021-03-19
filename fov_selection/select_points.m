function [phi theta]=select_points(img)
I = rgb2gray(img);
img = double(I);

%%% auto downsampling %%%
[M,N]=size(I);
f = max(1,round(min(M,N)/256));
if(f>1)
    lpf = ones(f,f);
    lpf = lpf/sum(lpf(:));
    img = imfilter(img,lpf,'symmetric','same');
    img = img(1:f:end,1:f:end);
end

%%% detect keypoints with padding %%%
img = [img(:,end-34:end,:) img(:,:,:) img(:,1:35,:)];
points = detectSURFFeatures(uint8(img));

%%% point map %%%
[m,n] = size(img);
point_map = zeros(m,n);
for i=1:length(points)
    point_map(round(points.Location(i,2)'),round(points.Location(i,1)'))=1;
end

%%% filter %%%
sigma = 10;
gausFilter = fspecial('gaussian', [71,71], sigma);
gaus_point = filter2(gausFilter, point_map, 'same');
gaus_point = gaus_point(:,36:end-35);
scale=255/max(max(gaus_point));
gaus_point = gaus_point*scale;

%%% select 20 point %%%
[H,W]=size(gaus_point);
distance = SphereDist([0;0],[pi/6;0]);
[M(1),idx(1)] = max(gaus_point(:));
[I_row(1), I_col(1)] = ind2sub(size(gaus_point),idx(1));
[phi(1) theta(1)] = erp2sph(I_col(1),I_row(1),W,H);
gaus_point(I_row(1), I_col(1))=0;
i=2;
while i<21
    [max_num,idx_num] = max(gaus_point(:));
    [idx_row, idx_col] = ind2sub(size(gaus_point),idx_num);
    [idx_phi, idx_theta] = erp2sph(idx_col(1),idx_row(1),W,H);
    gaus_point(idx_row, idx_col)=0;
    dist = min(SphereDist([idx_phi;idx_theta],[phi;theta]));
%     dist = min(sqrt((idx_row-I_row).^2 + (idx_col-I_col).^2));
    if dist > distance
        I_row(i) = idx_row;
        I_col(i) = idx_col;
        phi(i) = idx_phi;
        theta(i) = idx_theta;
        i = i + 1;
    end
end    
    
    








