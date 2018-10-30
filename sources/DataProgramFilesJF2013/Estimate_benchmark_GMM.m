function [results]=Estimate_benchmark_GMM(Y,X_1,X_2)

% Last modified: 05-15-2012
%
% Estimates the predictive regression model,
%
% r(i,t+1) = b(i,0) + b(i,1)*x(i,1,t) + b(i,2)*x(i,2,t) + e(i,t+1),
%
% where test statistics are computed using a GMM procedure.
%
% Input
%
% Y   = T-by-N matrix of excess return observations
% X_1 = T-by-N matrix of observations for first predictor
% X_2 = T-by-N matrix of observations for second predictor
%
% Output
%
% results = (N+1)-by-6 matrix;
%           First N rows: b(i,1) estimates, t-stats,
%                         b(i,2) estimates, t-stats,
%                         R-squared stats, chi-squared-stats
%           (N+1) row: pooled b(1) estimate, t-stat, 
%                      pooled b(2) estimate, t-stat,
%                      pooled R-squared stat, chi-squared-stat

%%%%%%%%%%%%%%%
% Preliminaries
%%%%%%%%%%%%%%%

[T,N]=size(Y);
K=3; % number of RHS variables (including intercept)
results=zeros(N+1,6);

%%%%%%%%%%%%%%%%%%%%%
% Setting up matrices
%%%%%%%%%%%%%%%%%%%%%

Y_stack=zeros(N*(T-1),1);
X_stack=zeros(N*(T-1),N*K);
for t=1:(T-1);
    for i=1:N;
        Y_stack((t-1)*N+i)=Y(t+1,i);
        X_stack((t-1)*N+i,(i-1)*K+1:i*K)=[X_1(t,i) X_2(t,i) 1];
    end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OLS estimation with GMM standard errors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

beta_ols=((X_stack'*X_stack)^(-1))*(X_stack'*Y_stack);
E_ols=Y_stack-X_stack*beta_ols;
h_ols=zeros(N*K,T-1);
for t=1:(T-1);
    h_ols(:,t)=(X_stack((t-1)*N+1:t*N,:))'*E_ols((t-1)*N+1:t*N);
end;
S_ols=(1/(T-1))*(h_ols*h_ols');
d_ols=zeros(N*K,N*K,T-1);
for t=1:(T-1);
    for i=1:N;
        x_i_t=X_stack((t-1)*N+i,(i-1)*K+1:i*K)';
        d_ols((i-1)*K+1:i*K,(i-1)*K+1:i*K,t)=x_i_t*x_i_t';
    end;
end;
D_ols=(1/(T-1))*sum(d_ols,3);
V_ols=(1/(T-1))*((D_ols'*(S_ols^(-1))*D_ols)^(-1));
SE_ols=sqrt(diag(V_ols));
t_stat_ols=beta_ols./SE_ols;
for i=1:N;
    results(i,1:4)=[beta_ols((i-1)*K+1) t_stat_ols((i-1)*K+1) ...
        beta_ols((i-1)*K+2) t_stat_ols((i-1)*K+2)];
end;
e=zeros(T-1,N);
for t=1:(T-1);
    for i=1:N;
        e(t,i)=E_ols((t-1)*N+i);
    end;
end;
for i=1:N;
    deviation_i=Y(2:end,i)-mean(Y(2:end,i));
    TSS_i=deviation_i'*deviation_i;
    RSS_i=e(:,i)'*e(:,i);
    results(i,5)=1-(RSS_i/TSS_i);
    R_i=zeros(2,N*K);
    R_i(:,((i-1)*K+1):((i-1)*K+(K-1)))=eye(K-1);
    results(i,6)=(R_i*beta_ols)'*((R_i*V_ols*R_i')^(-1))*(R_i*beta_ols);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pooled estimation with GMM standard errors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

C_pool=zeros((N-1)*(K-1),N*K);
for i=1:(N-1);
    C_pool((i-1)*(K-1)+1:i*(K-1),1:K)=[eye(K-1) zeros(K-1,1)];
    C_pool((i-1)*(K-1)+1:i*(K-1),i*K+1:(i+1)*K)=[-eye(K-1) zeros(K-1,1)];
end;
beta_pool=beta_ols-((X_stack'*X_stack)^(-1))*C_pool'*...
    ((C_pool*((X_stack'*X_stack)^(-1))*C_pool')^(-1))*C_pool*beta_ols;
E_pool=Y_stack-X_stack*beta_pool;
h_pool=zeros(N*K,T-1);
for t=1:(T-1);
    h_pool(:,t)=(X_stack((t-1)*N+1:t*N,:))'*E_pool((t-1)*N+1:t*N);
end;
h_bar_pool=mean(h_pool,2);
S_pool=(1/(T-1))*(h_pool*h_pool');
d_pool=zeros(N*K,N+(K-1),T-1);
for t=1:(T-1);
    for i=1:N;
        x_i_t=X_stack((t-1)*N+i,(i-1)*K+1:i*K)';
        xx_i_t=x_i_t*x_i_t';
        d_pool((i-1)*K+1:i*K,1:(K-1),t)=xx_i_t(:,1:(K-1));
        d_pool((i-1)*K+1:i*K,(K-1)+i,t)=xx_i_t(:,K);
    end;
end;
D_pool=(1/(T-1))*sum(d_pool,3);
V_pool=(1/(T-1))*((D_pool'*D_pool)^(-1))*(D_pool'*S_pool*D_pool)*((D_pool'*D_pool)^(-1));
SE_pool=sqrt(diag(V_pool));
beta_pool_intercept=zeros(N,1);
for i=1:N;
    beta_pool_intercept(i)=beta_pool(i*K);
end;
beta_pool=[beta_pool(1:(K-1)) ; beta_pool_intercept];
t_stat_pool=beta_pool./SE_pool;
results(N+1,1:4)=[beta_pool(1) t_stat_pool(1) beta_pool(2) ...
    t_stat_pool(2)];
Y_stack_deviation=Y_stack-mean(Y_stack);
TSS_pool=Y_stack_deviation'*Y_stack_deviation;
RSS_pool=E_pool'*E_pool;
results(N+1,5)=1-(RSS_pool/TSS_pool);
R_pool=[eye(K-1) zeros(2,N) ];
results(N+1,6)=(R_pool*beta_pool)'*((R_pool*V_pool*R_pool')^(-1))*...
    (R_pool*beta_pool);
