function error = fun(x,P, T, hiddennum,P_test,T_test)
inputnum=size(P,1);       % 输入层神经元个数
outputnum=size(T,1);      % 输出层神经元个数

% N = size(P_test,1); % 获取测试集的列数，即测试样本数

% 训练集归一化
[Ptrain,inFP] = mapminmax(P);
Ptest = mapminmax('apply',P_test,inFP);
% 测试集归一化
[Ttrain,outFP] = mapminmax(T);
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

error=norm(T_sim-T_test);

