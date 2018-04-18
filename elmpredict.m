function Y = elmpredict(P,IW,B,LW,TF,TYPE)

if nargin < 6   % ����Ӻ��������������6��������
    error('ELM:Arguments','Not enough input arguments.');
end

% Calculate the Layer Output Matrix H ������������H
Q = size(P,2);               % ��ȡ������������������
BiasMatrix = repmat(B,1,Q);  % ����������������������ƫ����󣨸��Ʒ�ʽ��
tempH = IW * P + BiasMatrix; % ��������������
% �ж�������ݺ���
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end

% Calculate the Simulate Output ����ģ�����
% ���㺯��F(iw*p+e)�Ľ��
Y = (H' * LW)';
if TYPE == 1
    temp_Y = zeros(size(Y)); 
    for i = 1:size(Y,2)
        [Max_Y,index] = max(Y(:,i));
        temp_Y(index,i) = 1;
    end
    Y = vec2ind(temp_Y); %��ʸ��ת��Ϊ����
end

% ELMPREDICT Simulate a Extreme Learning Machine
% Syntax
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Description˵��
% Input����
% P   - Input Matrix of Training Set (R*Q)ѵ������������� 
% IW  - Input Weight Matrix (N*R)����Ȩֵ
% B   - Bias Matrix  (N*1)ƫ��
% LW  - Layer Weight Matrix (N*S)ͼ��Ȩֵ
% TF  - Transfer Function:���ݺ���
%       'sig' for Sigmoidal function (default)S�ͺ���
%       'sin' for Sine function���Һ���
%       'hardlim' for Hardlim functionӲ�����ʹ��ݺ���
% TYPE - Regression (0,default) or Classification (1) �ع飨0��Ĭ��ֵ������ࣨ1��
% Output���
% Y   - Simulate Output Matrix (S*Q)ģ���������
% Example
% Regression:�ع�
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',0)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Classification:����
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',1)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)