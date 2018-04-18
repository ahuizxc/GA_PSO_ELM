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

% 参数初始化
%粒子群算法中的两个参数
c1 = 1.49445;
c2 = 1.49445;

maxgen=50;   % 进化次数  
sizepop=10;   %种群规模
hiddennum=40;
Vmax=1;
Vmin=-1;
popmax=5;
popmin=-5;

for i=1:sizepop
    pop(i,:)=5*rands(1,size(P,2)+11);
    V(i,:)=rands(1,size(P,2)+11);
    fitness(i)=fun(pop(i,:),P, T, hiddennum,P_test,T_test);
end


% 个体极值和群体极值
[bestfitness,bestindex]=min(fitness);
zbest=pop(bestindex,:);   %全局最佳
gbest=pop;    %个体最佳
fitnessgbest=fitness;   %个体最佳适应度值
fitnesszbest=bestfitness;   %全局最佳适应度值

%% 迭代寻优
for i=1:maxgen
     
    for j=1:sizepop
        
        %速度更新
        V(j,:) = V(j,:) + c1*rand*(gbest(j,:) - pop(j,:)) + c2*rand*(zbest - pop(j,:));
        V(j,find(V(j,:)>Vmax))=Vmax;
        V(j,find(V(j,:)<Vmin))=Vmin;
        
        %种群更新
        pop(j,:)=pop(j,:)+0.2*V(j,:);
        pop(j,find(pop(j,:)>popmax))=popmax;
        pop(j,find(pop(j,:)<popmin))=popmin;
        
        %自适应变异
        pos=unidrnd(size(P,2)+11);
        if rand>0.95
            pop(j,pos)=5*rands(1,1);
        end
      
        %适应度值
        fitness(j)=fun(pop(j,:),P, T, hiddennum,P_test,T_test);
    end
    
    for j=1:sizepop
        %个体最优更新
        if fitness(j) < fitnessgbest(j)
            gbest(j,:) = pop(j,:);
            fitnessgbest(j) = fitness(j);
        end
        
        %群体最优更新
        if fitness(j) < fitnesszbest
            zbest = pop(j,:);
            fitnesszbest = fitness(j);
        end
    
    end
    
    yy(i)=fitnesszbest;    
        
end

%% 结果分析
figure(2),
grid on
P0 =plot(yy);
title('PSO-ELM适应度曲线');
xlabel('进化代数');ylabel('适应度');
set(P0,'LineWidth',1.5);