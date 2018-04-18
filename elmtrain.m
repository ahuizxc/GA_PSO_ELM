% ELM ѵ������
function [LW,TF,TYPE] = elmtrain(P,T,N,TF,TYPE,IW,B);

% nargin : ���������������
if nargin < 2                
    error('ELM:Arguments','Not enough input arguments.');
end
% ���������������2�����趨Ĭ��ֵ
if nargin < 3  
    N = size(P,2);
end
% ���������������3��������Ĭ�Ϻ������趨Ĭ��ֵ
if nargin < 4 
    TF = 'sig';
end
% ���������������4�����趨�������� Regression (0,default) or Classification (1)���趨Ĭ��ֵ
if nargin < 5 
    TYPE = 0;
end   
% ���������������5�����趨����Ȩֵ
if nargin < 6 
    IW = rand(N,R) * 2 - 1;
end   
% ���������������6�����趨ƫ��
if nargin < 7 
    B = rand(N,1);
end   
% ~= �����ڣ������������Ƿ�һ��
if size(P,2) ~= size(T,2)  
   error('ELM:Arguments','The columns of P and T must be same.');
end
% ��ȡ����ѵ�����������к���
[R,Q] = size(P);
if TYPE  == 1        % ���TYPE=1ΪClassification ���ຯ��
    T  = ind2vec(T); % ͨ��ind2vec�������ܽ���0��1����
end
[S,Q] = size(T);     % ��ȡ���ѵ�����������к���
BiasMatrix = repmat(B,1,Q); % repmat(B,n1,n2);������B����n1*n2�� size(B,2)*n1 size(B,1)*n2

% Calculate the Layer Output Matrix H ������������H
tempH = IW * P + BiasMatrix;
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end
% Calculate the Output Weight Matrix ���������Ȩֵ����
LW = pinv(H') * T';


%  ���ע��
% ELMTRAIN Create and Train a Extreme Learning Machine
% Syntax �﷨
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,N,TF,TYPE)
% Description ����
% Input
% P   - Input Matrix of Training Set  (R*Q)  ѵ����������
% T   - Output Matrix of Training Set (S*Q) ѵ���������
% N   - Number of Hidden Neurons (default = Q) ������ڵ���
% TF  - Transfer Function: ���ݺ�����ת������
%       'sig' for Sigmoidal function (default) S�ͺ���
%       'sin' for Sine function ���Һ���
%       'hardlim' for Hardlim function Ӳ�����ʹ��ݺ���
% TYPE - Regression (0,default) or Classification (1)
% Output
% IW  - Input Weight Matrix (N*R) ����Ȩֵ
% B   - Bias Matrix  (N*1) ƫ��
% LW  - Layer Weight Matrix (N*S)
% Example
% Regression:
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',0)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Classification
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',1)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
