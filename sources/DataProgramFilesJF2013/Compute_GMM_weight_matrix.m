function W=Compute_GMM_weight_matrix(mom,nlag)

% Last modified: 05-25-2012

q=size(mom,2);
T=size(mom,1);
a2=zeros(q,q);
a3=zeros(q,q);
for j=1:nlag;
    a1=zeros(q,q);
    for i=1:(T-j)
        a1=mom(i+j,:)'*mom(i,:)+a1;
    end;
    S(:,:,j)=T/(T-q)*1/T*a1;
    a2=(1-j/(nlag+1))*S(:,:,j)+a2;
    a3=(1-j/(nlag+1))*S(:,:,j)'+a3;
end
b1=zeros(q,q);
for i=1:T;
    b1=mom(i,:)'*mom(i,:)+b1;
end;
if nlag==0;
    newS=b1*T/(T-q)*1/T;
else 
    newS=a2+a3+b1*T/(T-q)*1/T;
end;
W=pinv(newS);
