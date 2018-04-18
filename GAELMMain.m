clc;
clear all;
%% 加载神经网络的训练样本 测试样本每列一个样本 输入P 输出T
load 数据; % 载入数据
data = SOC';
data_size = size(data,2);
lin = ceil(linspace(1,data_size,300)); % 在样本空间均匀取值300组

train_id = linspace(1,299,150);   % 奇数组作为训练样本集
test_id = train_id+1;             % 偶数组作为预测样本集

P= data(2:3,lin(train_id));       %电压电流训练输入
T= data(4,lin(train_id));         % SOC作为训练输出

P_test = data(2:3,lin(test_id));  % 电压电流预测输入
T_test = data(4,lin(test_id));    % SOC作为预测输出
%%
hiddennum=40;                     % 初始隐层神经元个数
threshold= minmax(P) ;            % 输入向量的最大值和最小值
inputnum=size(P,1);               % 输入层神经元个数
outputnum=size(T,1);              % 输出层神经元个数
w1num=inputnum*hiddennum;         % 输入层到隐层的权值个数
w2num=outputnum*hiddennum;        % 隐层到输出层的权值个数
N=w1num+hiddennum+w2num+outputnum;% 待优化的变量的个数

%% 定义遗传算法参数
NIND=50;        % 个体数目(Number of individuals)
MAXGEN=50;     % 最大遗传代数(Maximum number of generations)
PRECI=30;       % 变量的二进制位数(Precision of variables)
GGAP=0.95;      % 代沟(Generation gap)
px=0.7;         % 交叉概率
pm=0.01;        % 变异概率
trace=zeros(N+1,MAXGEN); % 寻优结果的初始值
FieldD=[repmat(PRECI,1,N);repmat([-0.5;0.5],1,N);repmat([1;0;1;1],1,N)]; % 区域描述器(Build field descriptor)
Chrom=crtbp(NIND,PRECI*N); % 初始种群
%% 优化
gen=0;                                      % 代计数器
X=bs2rv(Chrom,FieldD);                      % 计算初始种群的十进制转换
ObjV=Objfun(X,P,T,hiddennum,P_test,T_test); % 计算目标函数值
while gen<MAXGEN
   fprintf('%d\n',gen)
   FitnV=ranking(ObjV);                  % 分配适应度值(Assign fitness values)
   SelCh=select('sus',Chrom,FitnV,GGAP); % 选择，sus为随机遍历抽样函数
   SelCh=recombin('xovsp',SelCh,px);     % 重组，xovsp单点交叉函数
   SelCh=mut(SelCh,pm);                  % 变异，mut二进制和整型变异函数
   X=bs2rv(SelCh,FieldD);                            % 子代个体的十进制转换
   ObjVSel=Objfun(X,P,T,hiddennum,P_test,T_test);    % 计算子代的目标函数值
   [Chrom,ObjV]=reins(Chrom,SelCh,1,1,ObjV,ObjVSel); % 重插入子代到父代，得到新种群
   X=bs2rv(Chrom,FieldD);
   gen=gen+1;              % 代计数器增加，由预设代数和定义种群的最优个体来决定终止
   
   [Y,I]=min(ObjV);        % 获取每代的最优解及其序号，Y为最优解,I为个体的序号
   trace(1:N,gen)=X(I,:);  % 记下每代的最优值
   trace(end,gen)=Y;       % 记下每代的最优值
end
%% 画进化图
figure(1);
P0 = plot(1:MAXGEN,trace(end,:));
grid on
xlabel('遗传代数')
ylabel('误差的变化')
title('GA-ELM优化过程')
bestX=trace(1:end-1,end);
bestErr=trace(end,end);
fprintf(['最优初始权值和阈值:\nX=',num2str(bestX'),'\n最小误差err=',num2str(bestErr),'\n'])
set(P0,'LineWidth',1.5);       % 设置图形线宽

%%
PSO
%% 比较优化前后的训练&测试
callELMfun