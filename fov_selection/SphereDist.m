function d = SphereDist(x,y,R)
%��������ľ�γ�ȼ����Բ����(�����������ҹ�ʽ)
%xΪA��[����, γ��], yΪB��[����, γ��]
if nargin < 3
    R = 1;
end

DeltaS = acos(cos(x(2)).*cos(y(2,:)).*cos(x(1)-y(1,:))+sin(x(2)).*sin(y(2,:)));
d = R*DeltaS;