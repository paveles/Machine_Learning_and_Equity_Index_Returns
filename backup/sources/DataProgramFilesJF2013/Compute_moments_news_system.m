function f=Compute_moments_news_system(parameters,num,Y_USA,Y,X_USA,X)

% Last modified: 05-25-2012

T=size(Y,1);
N=size(Y,2)+1;
beta_USA=parameters(1:size(X_USA,2));
u_USA=[0 ; Y_USA(2:end)-X_USA(1:end-1,:)*beta_USA];
U(:,1)=u_USA;
for i=1:N-1;
    Y_i=Y(:,i);
    X_i=X(:,:,i);
    beta_i=parameters(size(X_USA,2)+(size(X_i,2)+2)*(i-1)+1:size(X_USA,2)+(size(X_i,2)+2)*i);
    u_i=Y_i(2:end)-(X_i(1:end-1,:)*beta_i(1:size(X_i,2))+(beta_i(end-1)*beta_i(end))*u_USA(2:end)+...
        (beta_i(end-1)*(1-beta_i(end)))*u_USA(1:end-1));
    u_i=[0 ; u_i];
    U(:,i+1)=u_i;
end;
K=size(X_USA,2)+(size(X,2)-1)*(N-1)+(size(X,2)+1)*(N-1)+(N-1);
for t=2:T;
    m_t_USA=[kron([X_USA(t-1,:)],U(t,1))];
    m_t_i=[];
    for i=1:N-1;
        m_t_i=[m_t_i kron(X(t-1,2:end,i),U(t,1)) kron([X(t-1,:,i) U(t-1,1)],U(t,1+i)) U(t,1)*U(t,1+i)];
    end;
    m_t(t,:)=[m_t_USA m_t_i];
end;
m_t=m_t(2:end,:);
m=mean(m_t)';
if num==1;
    f=m_t;
elseif num==2;
    f=m;
end;
