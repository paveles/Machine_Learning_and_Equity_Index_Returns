function[qLL_hat]=Compute_qLL_hat(y,X,Z,L)

% Last modified: 08-18-2015

% Computes the Elliott and Muller (2006) statistic for testing the null
% that beta(t) = beta for all t.
%
% Input
%
% y = T-vector of dependent variable observations
% X = T-by-k data matrix linked with potentially changing coefficients
% Z = T-by-d data matrix linked with constant coefficients (empty matrix
%     if all coefficients allowed to change)
% L = lag truncation for Newey-West estimator
%
% Output
%
% qLL_hat = Elliott and Muller (2006) statistic (see Table 1 of paper for
%           critical values)
%
% Reference
%
% Elliott, G and UK Muller, (2006), "Efficient Tests for General Persistent
% Time Variation in Regression Coefficients," Review of Economic Studies
% 73, 907-940

[T,k]=size(X);

% Step 1

results=ols(y,[X Z]);
epsilon_hat=results.resid;

% Step 2

X_epsilon_hat=X.*kron(ones(1,k),epsilon_hat);
V_hat_X=(1/T)*(X_epsilon_hat'*X_epsilon_hat);
if L>=1;
    for l=1:L;
        Gamma_hat_l=(1/T)*...
            (X_epsilon_hat(l+1:end,:)'*X_epsilon_hat(1:end-l,:));
        V_hat_X=V_hat_X+(1-(l/(L+1)))*(Gamma_hat_l+Gamma_hat_l');
    end;
end;

% Step 3

U_hat=(V_hat_X^(-0.5))*X_epsilon_hat';

% Step 4

r_bar=1-(10/T);
w_hat=zeros(k,T);
for t=1:T;
    if t==1;
        w_hat(:,t)=U_hat(:,t);
    else
        w_hat(:,t)=r_bar*w_hat(:,t-1)+(U_hat(:,t)-U_hat(:,t-1));
    end;
end;

% Step 5

r_bar_trend=r_bar.^((1:1:length(y))');
SSR=nan(k,1);
for i=1:k;
    results_i=ols(w_hat(i,:)',r_bar_trend);
    SSR(i)=results_i.resid'*results_i.resid;
end;

% Step 6

qLL_hat=r_bar*sum(SSR)-sum(sum(U_hat.^2,2));
