function err=ELMfun(x,P_train,T_train,hiddennum,P_test,T_test)

%% ѵ��&����ELM����
%% ����
% x��һ������ĳ�ʼȨֵ����ֵ
% P��ѵ����������
% T��ѵ���������
% hiddennum����������Ԫ��
% P_test:������������
% T_test:���������������
%% ���
% err��Ԥ��������Ԥ�����ķ���

inputnum=size(P_train,1);       % �������Ԫ����
outputnum=size(T_train,1);      % �������Ԫ����

% N = size(P_test,1); % ��ȡ���Լ���������������������

% ѵ������һ��
[Ptrain,inFP] = mapminmax(P_train);
Ptest = mapminmax('apply',P_test,inFP);
% ���Լ���һ��
[Ttrain,outFP] = mapminmax(T_train);
Ttest = mapminmax('apply',T_test,outFP);

%% ELM��ʼȨֵ����ֵ
w1num=inputnum*hiddennum; % ����㵽�����Ȩֵ����
w1=x(1:w1num);   %��ʼ����㵽�����Ȩֵ
w1 = reshape(w1,hiddennum,inputnum);
B1=x(w1num+1:w1num+hiddennum);  %��ʼ������ֵ
B1=reshape(B1,hiddennum,1);

%% ELM ѵ��
% ����ELM����
[LW,TF,TYPE] = elmtrain(Ptrain,Ttrain,hiddennum,'sig',0,w1,B1);
% ELM�������
Tsim = elmpredict(Ptest,w1,B1,LW,TF,TYPE);
% ����һ��
T_sim = mapminmax('reverse',Tsim,outFP);

err=norm(T_sim-T_test);

% ����Ա�
% result = [T_test' T_sim'];
% err=norm(Y-T_test);
% �������  abs(����1-����2).^2/������
% E = mse(T_sim - T_test); 
% ����ϵ��
% N = length(T_test);
% R2=(N*sum(T_sim.*T_test)-sum(T_sim)*sum(T_test))^2/((N*sum((T_sim).^2)-(sum(T_sim))^2)*(N*sum((T_test).^2)-(sum(T_test))^2)); 

