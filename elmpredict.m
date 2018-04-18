function Y = elmpredict(P,IW,B,LW,TF,TYPE)

if nargin < 6   % 如果子函数输入参数少于6个，报错
    error('ELM:Arguments','Not enough input arguments.');
end

% Calculate the Layer Output Matrix H 计算输出层矩阵H
Q = size(P,2);               % 获取测试输入样本的行数
BiasMatrix = repmat(B,1,Q);  % 根据输入样本行数，扩充偏差矩阵（复制方式）
tempH = IW * P + BiasMatrix; % 计算测试输出矩阵
% 判断输出传递函数
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end

% Calculate the Simulate Output 计算模拟输出
% 计算函数F(iw*p+e)的结果
Y = (H' * LW)';
if TYPE == 1
    temp_Y = zeros(size(Y)); 
    for i = 1:size(Y,2)
        [Max_Y,index] = max(Y(:,i));
        temp_Y(index,i) = 1;
    end
    Y = vec2ind(temp_Y); %将矢量转换为索引
end

% ELMPREDICT Simulate a Extreme Learning Machine
% Syntax
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Description说明
% Input输入
% P   - Input Matrix of Training Set (R*Q)训练集的输入矩阵 
% IW  - Input Weight Matrix (N*R)输入权值
% B   - Bias Matrix  (N*1)偏置
% LW  - Layer Weight Matrix (N*S)图层权值
% TF  - Transfer Function:传递函数
%       'sig' for Sigmoidal function (default)S型函数
%       'sin' for Sine function正弦函数
%       'hardlim' for Hardlim function硬限制型传递函数
% TYPE - Regression (0,default) or Classification (1) 回归（0，默认值）或分类（1）
% Output输出
% Y   - Simulate Output Matrix (S*Q)模拟输出矩阵
% Example
% Regression:回归
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',0)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Classification:分类
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',1)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)