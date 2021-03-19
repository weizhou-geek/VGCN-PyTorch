function im2fov(img,spoint,i)
[M,~,~]=size(img);
fov_size=round(M/2);
parfor k=1:length(spoint)
    img_fov=cut_patch(img,spoint(k,1),spoint(k,2),fov_size);
    imwrite(uint8(img_fov),[i,'_fov',num2str(k),'.png']);
end