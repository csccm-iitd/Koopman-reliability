clc
clear all
rng(1)
nrp=10000;
tspan=[0 3];
t_p=100;
% A=0.02+0.02*rand(nrp,1);
% B=2+4*rand(nrp,1);
% C=0.1+0.1*rand(nrp,1);
% D=2+6*rand(nrp,1);

A=0.02+0.05*rand(nrp,1);
B=2+5*rand(nrp,1);
C=0.1+0.4*rand(nrp,1);
D=2+7*rand(nrp,1);

y_data1=zeros(nrp*t_p,2);
y_data2=zeros(nrp*t_p,2);
y_label=zeros(nrp*t_p,4);
t_data=zeros(nrp*t_p,1);
y0=[1  1];
j=0;
for i=1:nrp
    a=A(i);
    b=B(i);
    c=C(i);
    d=D(i);
    sol=ode45(@(t,y)duffing(t,y,a,b,c,d),tspan,y0);
    t=linspace(0,3,t_p+1)';
    y=deval(sol,t)';
    y_data1(j+1:j+t_p,:)=y(1:t_p,:);
    y_data2(j+1:j+t_p,:)=y(2:t_p+1,:);
    label=repmat([a,b,c,d],t_p,1);
    y_label(j+1:j+t_p,:)=label;
    j=t_p*i;
end

label=y_label;

mat011u=y_data1;
mat012u=y_data2;
mat013u=y_label;
save data1u.mat mat011u mat012u mat013u -v7.3

t_d=array2table(t);
writetable(t_d,'t_data.csv')