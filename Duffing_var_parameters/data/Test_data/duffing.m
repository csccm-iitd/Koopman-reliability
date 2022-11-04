function dydt=duffing(t,y,a,b,c,d)
dydt=zeros(2,1);
dydt(1)=y(2);
dydt(2)=-a*y(2)-b*y(1)-c*y(1)^3-d*cos(t);
end