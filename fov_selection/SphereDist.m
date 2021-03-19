function d = SphereDist(x,y,R)
%根据两点的经纬度计算大圆距离(基于球面余弦公式)
%x为A点[经度, 纬度], y为B点[经度, 纬度]
if nargin < 3
    R = 1;
end

DeltaS = acos(cos(x(2)).*cos(y(2,:)).*cos(x(1)-y(1,:))+sin(x(2)).*sin(y(2,:)));
d = R*DeltaS;