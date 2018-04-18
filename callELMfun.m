clc
%% 不使用遗传算法
% 训练集归一化
[Ptrain,inFP] = mapminmax(P); % 通过将行最小值和最大值映射到[-1 1]来处理矩阵
Ptest = mapminmax('apply',P_test,inFP);
% 测试集归一化
[Ttrain,outFP] = mapminmax(T);
Ttest = mapminmax('apply',T_test,outFP);
%% ELM 训练
[IW,B,LW,TF,TYPE] = elmtrain2(Ptrain,Ttrain,30,'sig',0); % 训练ELM模型

%% 测试网络
disp('[1.优化前测试样本预测]') % 打印全部值
% ELM仿真测试
T_test_sim1 = elmpredict(Ptest,IW,B,LW,TF,TYPE);
T_train_sim1 = elmpredict(P,IW,B,LW,TF,TYPE);
% 反归一化
Y11 = mapminmax('reverse',T_test_sim1,outFP); % 输出测试样本
err1=norm(Y11-T_test);                        % 测试样本的仿真误差
disp(['测试样本仿真相对误差:',num2str(err1)])

%% 使用遗传算法
%% 使用优化后的权值和阈值
nputnum=size(P,1);       % 输入层神经元个数 
outputnum=size(T,1);     % 输出层神经元个数
% 训练集归一化
[Ptrain,inFP] = mapminmax(P);
Ptest = mapminmax('apply',P_test,inFP);
% 测试集归一化
[Ttrain,outFP] = mapminmax(T);
Ttest = mapminmax('apply',T_test,outFP);

%% elm初始权值和阈值

w1num=inputnum*hiddennum;           % 输入层到隐层的权值个数
w1=bestX(1:w1num);                  % 初始输入层到隐层的权值
B1=bestX(w1num+1:w1num+hiddennum);  % 初始隐层阈值
IW1=reshape(w1,hiddennum,inputnum); % 重塑阵列
IB1=reshape(B1,hiddennum,1);

%% 创建ELM网络
[LW,TF,TYPE] = elmtrain(Ptrain,Ttrain,hiddennum,'sig',0,IW1,IB1); % 训练ELM模型

%% 测试网络
disp('[2.GA优化后测试样本预测]')
% ELM仿真测试
T_test_sim2 = elmpredict(Ptest,IW1,IB1,LW,TF,TYPE);
T_train_sim2 = elmpredict(P,IW1,IB1,LW,TF,TYPE);
% 反归一化
Y21 = mapminmax('reverse',T_test_sim2,outFP); % 输出测试样本
err2=norm(Y21-T_test);                        % 测试样本的仿真误差
disp(['测试样本仿真相对误差:',num2str(err2)])
%%
%% elm初始权值和阈值
bestX = zbest';
w1num=inputnum*hiddennum;           % 输入层到隐层的权值个数
w1=bestX(1:w1num);                  % 初始输入层到隐层的权值
B1=bestX(w1num+1:w1num+hiddennum);  % 初始隐层阈值
IW1=reshape(w1,hiddennum,inputnum); % 重塑阵列
IB1=reshape(B1,hiddennum,1);

%% 创建ELM网络
[LW,TF,TYPE] = elmtrain(Ptrain,Ttrain,hiddennum,'sig',0,IW1,IB1); % 训练ELM模型

%% 测试网络
disp('[2.PSO优化后测试样本预测]')
% ELM仿真测试
T_test_sim2 = elmpredict(Ptest,IW1,IB1,LW,TF,TYPE);
T_train_sim2 = elmpredict(P,IW1,IB1,LW,TF,TYPE);
% 反归一化
Y31 = mapminmax('reverse',T_test_sim2,outFP); % 输出测试样本
err3=norm(Y31-T_test);                        % 测试样本的仿真误差
disp(['测试样本仿真相对误差:',num2str(err3)])

%% 误差分析
result = [T_test' Y21'];
N = length(T_test);
% 均方误差计算
rate0 = T_test(1,:);
rate1 = Y11(1,:);
rate2 = Y21(1,:);
rate3 = Y31(1,:);
% 均方误差 abs(参数1-参数2).^2/样本数
Er1 = mse(rate1-rate0); 
Er2 = mse(rate2-rate0);
Er3 = mse(rate3-rate0);
% 决定系数
Rr1=(N*sum(rate1.*rate0)-sum(rate1)*sum(rate0))^2/((N*sum((rate1).^2)-(sum(rate1))^2)*(N*sum((rate0).^2)-(sum(rate0))^2)); 
Rr2=(N*sum(rate2.*rate0)-sum(rate2)*sum(rate0))^2/((N*sum((rate2).^2)-(sum(rate2))^2)*(N*sum((rate0).^2)-(sum(rate0))^2)); 
Rr3=(N*sum(rate3.*rate0)-sum(rate3)*sum(rate0))^2/((N*sum((rate3).^2)-(sum(rate3))^2)*(N*sum((rate0).^2)-(sum(rate0))^2)); 

%% 绘图
% 曲线图
figure(3) % GA优化ELM预测曲线
P1=plot(1:N,rate0,'r-*',1:N,rate2,'b:o',1:N,rate3,'g--o', 1:N,rate1,'k--*');
grid on
legend('真实值','GA-ELM预测值','PSO-ELM预测值''ELM预测值')
xlabel('样本编号')
ylabel('样本数据')
string = {'测试集--预测结果对比(真实值,GA-ELM,PSO-ELM,ELM)';['ELM:(mse = ' num2str(Er1) ' R^2 = ' num2str(Rr1) ')'];['PSO-ELM:(mse = ' num2str(Er3) ' R^2 = ' num2str(Rr3) ')'];['GA-ELM:(mse = ' num2str(Er2) ' R^2 = ' num2str(Rr2) ')']};
% h1=axes('position',[0.2 0.3 0.5 0.5]);
title(string)

figure(4) % 误差对比
P1=plot(1:N,abs(rate0-rate2),'r-*',1:N,abs(rate0-rate3),'g-.*',1:N,abs(rate0-rate1),'b:o');
grid on
legend('GA-ELM误差','PSO-ELM误差','ELM误差')
xlabel('样本编号')
ylabel('样本误差')
string = {'测试集--预测结果对比(GA-ELM,PSO-ELM,ELM)'};
title(string)
set(P1,'LineWidth',1.5);       % 设置图形线宽



% figure(4) % 未优化ELM预测曲线
% plot(1:N,T_test,'r-*',1:N,Y11,'b:o') 
% grid on
% legend('真实值','ELM预测值') 
% xlabel('样本编号') 
% ylabel('样本数据') 
% string ={'测试集预测结果对比(ELM)';['ELM:(mse = ' num2str(Er1) ' R^2 = ' num2str(Rr1) ')']}; 
% title(string)