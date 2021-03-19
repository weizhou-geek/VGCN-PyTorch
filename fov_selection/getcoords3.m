function coords = getcoords3(lon,lat,anglex,angley)
faceSizex = 2*tan(anglex/2);
faceSizey = 2*tan(angley/2);
x = cos(lat)*cos(lon);
y = cos(lat)*sin(lon);
z = sin(lat);
%thetapoint=[a;b];
point=[x;y;z];
%tangentvector = [-b;a];

vector1 = [-sin(lon);cos(lon);0];
vector2 = [sin(lat)*cos(lon);sin(lat)*sin(lon);-cos(lat)];
coords = zeros(3,4);
coords(:,1)=point -vector1*faceSizex/2-vector2*faceSizey/2;
coords(:,2)=point +vector1*faceSizex/2-vector2*faceSizey/2;
coords(:,3)=point -vector1*faceSizex/2+vector2*faceSizey/2;
coords(:,4)=point +vector1*faceSizex/2+vector2*faceSizey/2;
end