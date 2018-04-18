function err=ELMfun(x,P_train,T_train,hiddennum,P_test,T_test)

%% 训练&测试ELM网络
%% 输入
% x：一个个体的初始权值和阈值
% P：训练样本输入
% T：训练样本输出
% hiddennum：隐含层神经元数
% P_test:测试样本输入
% T_test:测试样本期望输出
%% 输出
% err：预测样本的预测误差的范数

inputnum=size(P_train,1);       % 输入层神经元个数
outputnum=size(T_train,1);      % 输出层神经元个数

% N = size(P_test,1); % 获取测试集的列数，即测试样本数

% 训练集归一化
[Ptrain,inFP] = mapminmax(P_train);
Ptest = mapminmax('apply',P_test,inFP);
% 测试集归一化
[Ttrain,outFP] = mapminmax(T_train);
Ttest = mapminmax('apply',T_test,outFP);

%% ELM初始权值和阈值
w1num=inputnum*hiddennum; % 输入层到隐层的权值个数
w1=x(1:w1num);   %初始输入层到隐层的权值
w1 = reshape(w1,hiddennum,inputnum);
B1=x(w1num+1:w1num+hiddennum);  %初始隐层阈值
B1=reshape(B1,hiddennum,1);

%% ELM 训练
% 创建ELM网络
[LW,TF,TYPE] = elmtrain(Ptrain,Ttrain,hiddennum,'sig',0,w1,B1);
% ELM仿真测试
Tsim = elmpredict(Ptest,w1,B1,LW,TF,TYPE);
% 反归一化
T_sim = mapminmax('reverse',Tsim,outFP);

err=norm(T_sim-T_test);

% 结果对比
% result = [T_test' T_sim'];
% err=norm(Y-T_test);
% 均方误差  abs(参数1-参数2).^2/样本数
% E = mse(T_sim - T_test); 
% 决定系数
% N = length(T_test);
% R2=(N*sum(T_sim.*T_test)-sum(T_sim)*sum(T_test))^2/((N*sum((T_sim).^2)-(sum(T_sim))^2)*(N*sum((T_test).^2)-(sum(T_test))^2)); 

