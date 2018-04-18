clc
%% ��ʹ���Ŵ��㷨
% ѵ������һ��
[Ptrain,inFP] = mapminmax(P); % ͨ��������Сֵ�����ֵӳ�䵽[-1 1]���������
Ptest = mapminmax('apply',P_test,inFP);
% ���Լ���һ��
[Ttrain,outFP] = mapminmax(T);
Ttest = mapminmax('apply',T_test,outFP);
%% ELM ѵ��
[IW,B,LW,TF,TYPE] = elmtrain2(Ptrain,Ttrain,30,'sig',0); % ѵ��ELMģ��

%% ��������
disp('[1.�Ż�ǰ��������Ԥ��]') % ��ӡȫ��ֵ
% ELM�������
T_test_sim1 = elmpredict(Ptest,IW,B,LW,TF,TYPE);
T_train_sim1 = elmpredict(P,IW,B,LW,TF,TYPE);
% ����һ��
Y11 = mapminmax('reverse',T_test_sim1,outFP); % �����������
err1=norm(Y11-T_test);                        % ���������ķ������
disp(['������������������:',num2str(err1)])

%% ʹ���Ŵ��㷨
%% ʹ���Ż����Ȩֵ����ֵ
nputnum=size(P,1);       % �������Ԫ���� 
outputnum=size(T,1);     % �������Ԫ����
% ѵ������һ��
[Ptrain,inFP] = mapminmax(P);
Ptest = mapminmax('apply',P_test,inFP);
% ���Լ���һ��
[Ttrain,outFP] = mapminmax(T);
Ttest = mapminmax('apply',T_test,outFP);

%% elm��ʼȨֵ����ֵ

w1num=inputnum*hiddennum;           % ����㵽�����Ȩֵ����
w1=bestX(1:w1num);                  % ��ʼ����㵽�����Ȩֵ
B1=bestX(w1num+1:w1num+hiddennum);  % ��ʼ������ֵ
IW1=reshape(w1,hiddennum,inputnum); % ��������
IB1=reshape(B1,hiddennum,1);

%% ����ELM����
[LW,TF,TYPE] = elmtrain(Ptrain,Ttrain,hiddennum,'sig',0,IW1,IB1); % ѵ��ELMģ��

%% ��������
disp('[2.GA�Ż����������Ԥ��]')
% ELM�������
T_test_sim2 = elmpredict(Ptest,IW1,IB1,LW,TF,TYPE);
T_train_sim2 = elmpredict(P,IW1,IB1,LW,TF,TYPE);
% ����һ��
Y21 = mapminmax('reverse',T_test_sim2,outFP); % �����������
err2=norm(Y21-T_test);                        % ���������ķ������
disp(['������������������:',num2str(err2)])
%%
%% elm��ʼȨֵ����ֵ
bestX = zbest';
w1num=inputnum*hiddennum;           % ����㵽�����Ȩֵ����
w1=bestX(1:w1num);                  % ��ʼ����㵽�����Ȩֵ
B1=bestX(w1num+1:w1num+hiddennum);  % ��ʼ������ֵ
IW1=reshape(w1,hiddennum,inputnum); % ��������
IB1=reshape(B1,hiddennum,1);

%% ����ELM����
[LW,TF,TYPE] = elmtrain(Ptrain,Ttrain,hiddennum,'sig',0,IW1,IB1); % ѵ��ELMģ��

%% ��������
disp('[2.PSO�Ż����������Ԥ��]')
% ELM�������
T_test_sim2 = elmpredict(Ptest,IW1,IB1,LW,TF,TYPE);
T_train_sim2 = elmpredict(P,IW1,IB1,LW,TF,TYPE);
% ����һ��
Y31 = mapminmax('reverse',T_test_sim2,outFP); % �����������
err3=norm(Y31-T_test);                        % ���������ķ������
disp(['������������������:',num2str(err3)])

%% ������
result = [T_test' Y21'];
N = length(T_test);
% ����������
rate0 = T_test(1,:);
rate1 = Y11(1,:);
rate2 = Y21(1,:);
rate3 = Y31(1,:);
% ������� abs(����1-����2).^2/������
Er1 = mse(rate1-rate0); 
Er2 = mse(rate2-rate0);
Er3 = mse(rate3-rate0);
% ����ϵ��
Rr1=(N*sum(rate1.*rate0)-sum(rate1)*sum(rate0))^2/((N*sum((rate1).^2)-(sum(rate1))^2)*(N*sum((rate0).^2)-(sum(rate0))^2)); 
Rr2=(N*sum(rate2.*rate0)-sum(rate2)*sum(rate0))^2/((N*sum((rate2).^2)-(sum(rate2))^2)*(N*sum((rate0).^2)-(sum(rate0))^2)); 
Rr3=(N*sum(rate3.*rate0)-sum(rate3)*sum(rate0))^2/((N*sum((rate3).^2)-(sum(rate3))^2)*(N*sum((rate0).^2)-(sum(rate0))^2)); 

%% ��ͼ
% ����ͼ
figure(3) % GA�Ż�ELMԤ������
P1=plot(1:N,rate0,'r-*',1:N,rate2,'b:o',1:N,rate3,'g--o', 1:N,rate1,'k--*');
grid on
legend('��ʵֵ','GA-ELMԤ��ֵ','PSO-ELMԤ��ֵ''ELMԤ��ֵ')
xlabel('�������')
ylabel('��������')
string = {'���Լ�--Ԥ�����Ա�(��ʵֵ,GA-ELM,PSO-ELM,ELM)';['ELM:(mse = ' num2str(Er1) ' R^2 = ' num2str(Rr1) ')'];['PSO-ELM:(mse = ' num2str(Er3) ' R^2 = ' num2str(Rr3) ')'];['GA-ELM:(mse = ' num2str(Er2) ' R^2 = ' num2str(Rr2) ')']};
% h1=axes('position',[0.2 0.3 0.5 0.5]);
title(string)

figure(4) % ���Ա�
P1=plot(1:N,abs(rate0-rate2),'r-*',1:N,abs(rate0-rate3),'g-.*',1:N,abs(rate0-rate1),'b:o');
grid on
legend('GA-ELM���','PSO-ELM���','ELM���')
xlabel('�������')
ylabel('�������')
string = {'���Լ�--Ԥ�����Ա�(GA-ELM,PSO-ELM,ELM)'};
title(string)
set(P1,'LineWidth',1.5);       % ����ͼ���߿�



% figure(4) % δ�Ż�ELMԤ������
% plot(1:N,T_test,'r-*',1:N,Y11,'b:o') 
% grid on
% legend('��ʵֵ','ELMԤ��ֵ') 
% xlabel('�������') 
% ylabel('��������') 
% string ={'���Լ�Ԥ�����Ա�(ELM)';['ELM:(mse = ' num2str(Er1) ' R^2 = ' num2str(Rr1) ')']}; 
% title(string)