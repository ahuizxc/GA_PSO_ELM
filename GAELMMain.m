clc;
clear all;
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
%%
hiddennum=40;                     % ��ʼ������Ԫ����
threshold= minmax(P) ;            % �������������ֵ����Сֵ
inputnum=size(P,1);               % �������Ԫ����
outputnum=size(T,1);              % �������Ԫ����
w1num=inputnum*hiddennum;         % ����㵽�����Ȩֵ����
w2num=outputnum*hiddennum;        % ���㵽������Ȩֵ����
N=w1num+hiddennum+w2num+outputnum;% ���Ż��ı����ĸ���

%% �����Ŵ��㷨����
NIND=50;        % ������Ŀ(Number of individuals)
MAXGEN=50;     % ����Ŵ�����(Maximum number of generations)
PRECI=30;       % �����Ķ�����λ��(Precision of variables)
GGAP=0.95;      % ����(Generation gap)
px=0.7;         % �������
pm=0.01;        % �������
trace=zeros(N+1,MAXGEN); % Ѱ�Ž���ĳ�ʼֵ
FieldD=[repmat(PRECI,1,N);repmat([-0.5;0.5],1,N);repmat([1;0;1;1],1,N)]; % ����������(Build field descriptor)
Chrom=crtbp(NIND,PRECI*N); % ��ʼ��Ⱥ
%% �Ż�
gen=0;                                      % ��������
X=bs2rv(Chrom,FieldD);                      % �����ʼ��Ⱥ��ʮ����ת��
ObjV=Objfun(X,P,T,hiddennum,P_test,T_test); % ����Ŀ�꺯��ֵ
while gen<MAXGEN
   fprintf('%d\n',gen)
   FitnV=ranking(ObjV);                  % ������Ӧ��ֵ(Assign fitness values)
   SelCh=select('sus',Chrom,FitnV,GGAP); % ѡ��susΪ���������������
   SelCh=recombin('xovsp',SelCh,px);     % ���飬xovsp���㽻�溯��
   SelCh=mut(SelCh,pm);                  % ���죬mut�����ƺ����ͱ��캯��
   X=bs2rv(SelCh,FieldD);                            % �Ӵ������ʮ����ת��
   ObjVSel=Objfun(X,P,T,hiddennum,P_test,T_test);    % �����Ӵ���Ŀ�꺯��ֵ
   [Chrom,ObjV]=reins(Chrom,SelCh,1,1,ObjV,ObjVSel); % �ز����Ӵ����������õ�����Ⱥ
   X=bs2rv(Chrom,FieldD);
   gen=gen+1;              % �����������ӣ���Ԥ������Ͷ�����Ⱥ�����Ÿ�����������ֹ
   
   [Y,I]=min(ObjV);        % ��ȡÿ�������Ž⼰����ţ�YΪ���Ž�,IΪ��������
   trace(1:N,gen)=X(I,:);  % ����ÿ��������ֵ
   trace(end,gen)=Y;       % ����ÿ��������ֵ
end
%% ������ͼ
figure(1);
P0 = plot(1:MAXGEN,trace(end,:));
grid on
xlabel('�Ŵ�����')
ylabel('���ı仯')
title('GA-ELM�Ż�����')
bestX=trace(1:end-1,end);
bestErr=trace(end,end);
fprintf(['���ų�ʼȨֵ����ֵ:\nX=',num2str(bestX'),'\n��С���err=',num2str(bestErr),'\n'])
set(P0,'LineWidth',1.5);       % ����ͼ���߿�

%%
PSO
%% �Ƚ��Ż�ǰ���ѵ��&����
callELMfun