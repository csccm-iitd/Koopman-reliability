function dydt=lorenz(y,a,b,c)
dydt=zeros(3,1);
dydt(1)=-a*y(1)+a*y(2);
dydt(2)=b*y(1)-y(2)-y(1)*y(3);
dydt(3)=-c*y(3)+y(1)*y(2);
end