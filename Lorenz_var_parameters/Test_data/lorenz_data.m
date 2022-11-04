clear all
clc
rng(1)
a=28;
b=10;
c=2.6667;
tspan=[0 1];
n_bc=10000;
t_step=100;
%y_bc=[0+20*rand(n_bc,1),0+20*rand(n_bc,1),-10+20*rand(n_bc,1)];
%y_bc=[0+20*rand(n_bc,1),0+20*rand(n_bc,1),-10+20*rand(n_bc,1)];%better
y_bc=[-2+22*rand(n_bc,1),-2+22*rand(n_bc,1),-12+22*rand(n_bc,1)];%better

j=0;
y_cnct1=zeros(n_bc*t_step,3);
y_cnct2=zeros(n_bc*t_step,3);
for i=1:n_bc
   y0=y_bc(i,:);
   sol=ode45(@(t,y) lorenz(y,a,b,c),tspan,y0);
   t=linspace(0,1,t_step+1)';%Important
   y=deval(sol,t)';
   y_cnct1(j+1:j+t_step,:)=y(1:t_step,:);
   y_cnct2(j+1:j+t_step,:)=y(2:t_step+1,:);
   j=t_step*i;
end

mat011u=y_cnct1;
mat012u=y_cnct2;
save data1u.mat mat011u mat012u -v7.3

t_d=array2table(t);
writetable(t_d,'t_data.csv')
