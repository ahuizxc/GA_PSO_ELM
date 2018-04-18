%% �����������ѵ������ ��������ÿ��һ������ ����P ���T
load ����; % ��������
data = SOC';
data_size = size(data,2);
lin = ceil(linspace(1,data_size,300)); % �������ռ����ȡֵ300��

train_id = linspace(1,299,150);   % ��������Ϊѵ��������
test_id = train_id+1;             % ż������ΪԤ��������

P= data(2:3,lin(train_id));       %��ѹ����ѵ������
T= data(4,lin(train_id));         % SOC��Ϊѵ�����

P_test = data(2:3,lin(test_id));  % ��ѹ����Ԥ������
T_test = data(4,lin(test_id));    % SOC��ΪԤ�����

% ������ʼ��
%����Ⱥ�㷨�е���������
c1 = 1.49445;
c2 = 1.49445;

maxgen=50;   % ��������  
sizepop=10;   %��Ⱥ��ģ
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


% ���弫ֵ��Ⱥ�弫ֵ
[bestfitness,bestindex]=min(fitness);
zbest=pop(bestindex,:);   %ȫ�����
gbest=pop;    %�������
fitnessgbest=fitness;   %���������Ӧ��ֵ
fitnesszbest=bestfitness;   %ȫ�������Ӧ��ֵ

%% ����Ѱ��
for i=1:maxgen
     
    for j=1:sizepop
        
        %�ٶȸ���
        V(j,:) = V(j,:) + c1*rand*(gbest(j,:) - pop(j,:)) + c2*rand*(zbest - pop(j,:));
        V(j,find(V(j,:)>Vmax))=Vmax;
        V(j,find(V(j,:)<Vmin))=Vmin;
        
        %��Ⱥ����
        pop(j,:)=pop(j,:)+0.2*V(j,:);
        pop(j,find(pop(j,:)>popmax))=popmax;
        pop(j,find(pop(j,:)<popmin))=popmin;
        
        %����Ӧ����
        pos=unidrnd(size(P,2)+11);
        if rand>0.95
            pop(j,pos)=5*rands(1,1);
        end
      
        %��Ӧ��ֵ
        fitness(j)=fun(pop(j,:),P, T, hiddennum,P_test,T_test);
    end
    
    for j=1:sizepop
        %�������Ÿ���
        if fitness(j) < fitnessgbest(j)
            gbest(j,:) = pop(j,:);
            fitnessgbest(j) = fitness(j);
        end
        
        %Ⱥ�����Ÿ���
        if fitness(j) < fitnesszbest
            zbest = pop(j,:);
            fitnesszbest = fitness(j);
        end
    
    end
    
    yy(i)=fitnesszbest;    
        
end

%% �������
figure(2),
grid on
P0 =plot(yy);
title('PSO-ELM��Ӧ������');
xlabel('��������');ylabel('��Ӧ��');
set(P0,'LineWidth',1.5);