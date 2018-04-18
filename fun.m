function error = fun(x,P, T, hiddennum,P_test,T_test)
inputnum=size(P,1);       % �������Ԫ����
outputnum=size(T,1);      % �������Ԫ����

% N = size(P_test,1); % ��ȡ���Լ���������������������

% ѵ������һ��
[Ptrain,inFP] = mapminmax(P);
Ptest = mapminmax('apply',P_test,inFP);
% ���Լ���һ��
[Ttrain,outFP] = mapminmax(T);
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

error=norm(T_sim-T_test);

