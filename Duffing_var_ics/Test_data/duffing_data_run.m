clear all
clc
tspan=[0 3];
n_bc=10000;
t_step=100;
rng(2)
%y_bc=[-3+10*rand(n_bc,1),-3+10*rand(n_bc,1)]; more accurate results
y_bc=[-6+12*rand(n_bc,1),0+12*rand(n_bc,1)];
%y_bc =[-5+10*rand(n_bc,1),0+10*rand(n_bc,1)];
j=0;
y_cnct1=zeros(n_bc*t_step,2);
y_cnct2=zeros(n_bc*t_step,2);
for i=1:n_bc
   y0=y_bc(i,:);
   t=linspace(0,3,t_step+1)';
   sol=ode45(@duffing,t,y0);
   y=deval(sol,t)';
   y_cnct1(j+1:j+t_step,:)=y(1:t_step,:);
   y_cnct2(j+1:j+t_step,:)=y(2:t_step+1,:);
   j=100*i;
end

mat011u=y_cnct1;
mat012u=y_cnct2;
save data1u.mat mat011u mat012u -v7.3

% save('y_data1.mat','y_cnct1')
% %dlmwrite('y_data1.txt',y_cnct1)
% y_d1=array2table(y_cnct1);
% writetable(y_d1,'y_data1.csv')
% 
% save('y_data2.mat','y_cnct2')
% %dlmwrite('y_data2.txt',y_cnct2)
% y_d2=array2table(y_cnct2);
% writetable(y_d2,'y_data2.csv')

t_d=array2table(t);
writetable(t_d,'t_data.csv')
