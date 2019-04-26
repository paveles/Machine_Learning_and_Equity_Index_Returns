function [results,R_squared]=Estimate_Granger_pairwise_GMM(Y,Y_exclude,X_1,X_2,j)

% Last modified: 05-18-2012

% The function estimates the augmented predictive regression model,
%
% r(i,t+1) = b(i,0) + b(i,1)*x(i,1,t) + b(i,2)*x(i,2,t) + b(i,i)*r(i,t) +
%            b(i,j)*r(j,t) + e(i,t+1),
%
% where test statistics are computed using a GMM-based procedure
% Input
%
% Y         = T-by-N matrix of excess return observations
% Y_exclude = T-by-N matrix of excess return observations excluding
%             last day (set to Y if not a concern)
% X_1       = T-by-N matrix of observations for first predictor
% X_2       = T-by-N matrix of observations for second predictor
% j         = scalar country index
%
% Output
%
% results   = N-by-2 matrix of b(i,j) estimates, t-stats;
%             last row is pooled b(i,j) estimate, t-stat
%
% R_squared = N-vector of R-squared stats; last row is pooled R-squared

%%%%%%%%%%%%%%%
% Preliminaries
%%%%%%%%%%%%%%%

[T,N]=size(Y);
K=5; % number of RHS variables (including intercept)
results=zeros(N,2);
R_squared=zeros(N,1);

%%%%%%%%%%%%%%%%%%%%%
% Setting up matrices
%%%%%%%%%%%%%%%%%%%%%

Y_stack=zeros((N-1)*(T-1),1);
X_stack=zeros((N-1)*(T-1),(N-1)*K);
Y_no_j=Y;
Y_no_j(:,j)=[];
X_1_no_j=X_1;
X_1_no_j(:,j)=[];
X_2_no_j=X_2;
X_2_no_j(:,j)=[];
for t=1:(T-1);
    for i=1:(N-1);
        Y_stack((t-1)*(N-1)+i)=Y_no_j(t+1,i);
        if (j==1) || (j==6);
            X_stack((t-1)*(N-1)+i,(i-1)*K+1:i*K)=[Y(t,j) Y_no_j(t,i) ...
                X_1_no_j(t,i) X_2_no_j(t,i) 1];
        elseif j==2;
            if i==10;
                X_stack((t-1)*(N-1)+i,(i-1)*K+1:i*K)=[Y(t,j) ...
                    Y_no_j(t,i) X_1_no_j(t,i) X_2_no_j(t,i) 1];
            else;
                X_stack((t-1)*(N-1)+i,(i-1)*K+1:i*K)=[Y_exclude(t,j) ...
                    Y_no_j(t,i) X_1_no_j(t,i) X_2_no_j(t,i) 1];
            end;
        elseif (j==3) || (j==5);
            if (i==1) || (i==5);
                X_stack((t-1)*(N-1)+i,(i-1)*K+1:i*K)=[Y_exclude(t,j) ...
                    Y_no_j(t,i) X_1_no_j(t,i) X_2_no_j(t,i) 1];
            else;
                X_stack((t-1)*(N-1)+i,(i-1)*K+1:i*K)=[Y(t,j) ...
                    Y_no_j(t,i) X_1_no_j(t,i) X_2_no_j(t,i) 1];
            end;
        elseif j==4;
            if (i==2) || (i==10);
                X_stack((t-1)*(N-1)+i,(i-1)*K+1:i*K)=[Y(t,j) ...
                    Y_no_j(t,i) X_1_no_j(t,i) X_2_no_j(t,i) 1];
            else;
                X_stack((t-1)*(N-1)+i,(i-1)*K+1:i*K)=[Y_exclude(t,j) ...
                    Y_no_j(t,i) X_1_no_j(t,i) X_2_no_j(t,i) 1];
            end;
        elseif (j==7) || (j==8) || (j==9) || (j==10);
            if (i==1) || (i==6);
                X_stack((t-1)*(N-1)+i,(i-1)*K+1:i*K)=[Y_exclude(t,j) ...
                    Y_no_j(t,i) X_1_no_j(t,i) X_2_no_j(t,i) 1];
            else;
                X_stack((t-1)*(N-1)+i,(i-1)*K+1:i*K)=[Y(t,j) ...
                    Y_no_j(t,i) X_1_no_j(t,i) X_2_no_j(t,i) 1];
            end;
        elseif j==11;
            if (i==2);
                X_stack((t-1)*(N-1)+i,(i-1)*K+1:i*K)=[Y(t,j) ...
                    Y_no_j(t,i) X_1_no_j(t,i) X_2_no_j(t,i) 1];
            else;
                X_stack((t-1)*(N-1)+i,(i-1)*K+1:i*K)=[Y_exclude(t,j) ...
                    Y_no_j(t,i) X_1_no_j(t,i) X_2_no_j(t,i) 1];
            end;
        end;
    end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OLS estimation with GMM standard errors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

beta_ols=((X_stack'*X_stack)^(-1))*(X_stack'*Y_stack);
E_ols=Y_stack-X_stack*beta_ols;
h_ols=zeros((N-1)*K,T-1);
for t=1:(T-1);
    h_ols(:,t)=(X_stack((t-1)*(N-1)+1:t*(N-1),:))'*...
        E_ols((t-1)*(N-1)+1:t*(N-1));
end;
S_ols=(1/(T-1))*(h_ols*h_ols');
d_ols=zeros((N-1)*K,(N-1)*K,T-1);
for t=1:(T-1);
    for i=1:(N-1);
        x_i_t=X_stack((t-1)*(N-1)+i,(i-1)*K+1:i*K)';
        d_ols((i-1)*K+1:i*K,(i-1)*K+1:i*K,t)=x_i_t*x_i_t';
    end;
end;
D_ols=(1/(T-1))*sum(d_ols,3);
V_ols=(1/(T-1))*((D_ols'*(S_ols^(-1))*D_ols)^(-1));
SE_ols=sqrt(diag(V_ols));
t_stat_ols=beta_ols./SE_ols;
for i=1:(N-1);
    results(i,:)=[beta_ols((i-1)*K+1) t_stat_ols((i-1)*K+1)];
end;
e_ols=zeros(T-1,N-1);
for t=1:(T-1);
    for i=1:(N-1);
        e_ols(t,i)=E_ols((t-1)*(N-1)+i);
    end;
end;
for i=1:(N-1);
    deviation_i=Y_no_j(2:end,i)-mean(Y_no_j(2:end,i));
    TSS_i=deviation_i'*deviation_i;
    RSS_i=e_ols(:,i)'*e_ols(:,i);
    R_squared(i)=1-(RSS_i/TSS_i);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pooled estimation with GMM standard errors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

C_pool=zeros((N-2)*(K-1),(N-1)*K);
for i=1:(N-2);
    C_pool((i-1)*(K-1)+1:i*(K-1),1:K-1)=eye(K-1);
    C_pool((i-1)*(K-1)+1:i*(K-1),i*K+1:i*K+(K-1))=-eye(K-1);
end;
beta_pool=beta_ols-((X_stack'*X_stack)^(-1))*C_pool'*...
    ((C_pool*((X_stack'*X_stack)^(-1))*C_pool')^(-1))*C_pool*beta_ols;
E_pool=Y_stack-X_stack*beta_pool;
h_pool=zeros((N-1)*K,T-1);
for t=1:(T-1);
    h_pool(:,t)=(X_stack((t-1)*(N-1)+1:t*(N-1),:))'*...
    E_pool((t-1)*(N-1)+1:t*(N-1));
end;
S_pool=(1/(T-1))*(h_pool*h_pool');
d_pool=zeros((N-1)*K,(K-1)+(N-1),T-1);
for t=1:(T-1);
    for i=1:(N-1);
        x_i_t=X_stack((t-1)*(N-1)+i,(i-1)*K+1:i*K)';
        xx_i_t=x_i_t*x_i_t';
        d_pool((i-1)*K+1:i*K,1:K-1,t)=xx_i_t(:,1:K-1);
        d_pool((i-1)*K+1:i*K,K-1+i,t)=x_i_t;
    end;
end;
D_pool=(1/(T-1))*sum(d_pool,3);
V_pool=(1/(T-1))*((D_pool'*D_pool)^(-1))*(D_pool'*S_pool*D_pool)*...
    ((D_pool'*D_pool)^(-1));
SE_pool=sqrt(diag(V_pool));
beta_pool_intercept=zeros(N-1,1);
for i=1:(N-1);
    beta_pool_intercept(i)=beta_pool(i*K);
end;
beta_pool=[beta_pool(1:K-1) ; beta_pool_intercept];
t_stat_pool=beta_pool./SE_pool;
results(end,:)=[beta_pool(1) t_stat_pool(1)];
deviation_pool=Y_stack-mean(Y_stack);
TSS_pool=deviation_pool'*deviation_pool;
RSS_pool=E_pool'*E_pool;
R_squared(N)=1-(RSS_pool/TSS_pool);
