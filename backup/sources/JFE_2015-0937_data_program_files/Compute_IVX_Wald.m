function [A_tilde_IVX_K,W_IVX_K,p_value]=Compute_IVX_Wald(y,X,K,M_n,beta)

% Last modified: 08-18-2015

% Computes the Kostakis, Magdalinos, and Stamatogiannis (2015) IVX-Wald
% statistic for a predictive regression.
%
% Input
%
% y    = T-vector of return observations
% X    = T-by-r data matrix of predictor observations
% K    = forecast horizon
% M_n  = bandwith parameter
% beta = scalar in (0,1); specify 'close' to one for best performance
%
% Output
%
% A_tilde_IVX_K = r-vector of coefficient estimates
% W_IVX_K       = IVX-Wald statistic
% p_value       = p-value for IVX-Wald statistic
%
% Reference
%
% Kostakis, A, T Magdalinos, and MP Stamatogiannis (2015), "Robust
% Econometric Inference for Stock Return Predictability," Review of
% Financial Studies 28, 1506-1553

% OLS estimation of predictive regression system

results_OLS_y=ols(y(2:end),[ones(length(y)-1,1) X(1:end-1,:)]);
epsilon_hat=results_OLS_y.resid;
U_hat=nan(length(y)-1,size(X,2));
for j=1:size(X,2);
    results_OLS_j=ols(X(2:end,j),[ones(length(y)-1,1) X(1:end-1,j)]);
    U_hat(:,j)=results_OLS_j.resid;
end;

% Compute short-run/long-run covariance matrices; see eqs (13)-(15)

Sigma_hat_ee=(1/length(epsilon_hat))*(epsilon_hat'*epsilon_hat);
Sigma_hat_eu=(1/length(epsilon_hat))*(epsilon_hat'*U_hat);
Sigma_hat_uu=(1/length(epsilon_hat))*(U_hat'*U_hat);
Omega_hat_uu=Sigma_hat_uu;
Omega_hat_eu=Sigma_hat_eu;
if M_n>0;
    Lambda_hat_uu=zeros(size(U_hat,2),size(U_hat,2));
    Lambda_hat_ue=zeros(size(U_hat,2),1);
    for h=1:M_n;
        Lambda_hat_uu=Lambda_hat_uu+(1-(h/(M_n+1)))*...
            (1/(length(U_hat)-1))*(U_hat(h+1:end,:)'*U_hat(1:end-h,:));
        Lambda_hat_ue=Lambda_hat_ue+(1-(h/(M_n+1)))*...
            (1/(length(U_hat)-1))*(U_hat(h+1:end,:)'*...
            epsilon_hat(1:end-h));
    end;
    Omega_hat_uu=Sigma_hat_uu+Lambda_hat_uu+Lambda_hat_uu';
    Omega_hat_eu=Sigma_hat_eu+Lambda_hat_ue';
end;

% Construct instruments; see eqs (4)/(5)

R_nz=1-(1/(length(y)-1)^beta);
d_X=X(2:end,:)-X(1:end-1,:);
d_X=[zeros(1,size(X,2)) ; d_X];
Z_tilde=nan(length(X),size(X,2)); % instrument matrix
Z_tilde(1,:)=zeros(1,size(X,2));
for t=2:length(X);
    Z_tilde(t,:)=R_nz*Z_tilde(t-1,:)+d_X(t,:);
end;

% Construct cumulative variables; see Sec 5.1

y_K=nan(length(y)-(K-1),1);
X_K=nan(length(y)-(K-1),size(X,2));
Z_tilde_K=nan(length(y)-(K-1),size(X,2));
for t=1:length(y)-(K-1);
    y_K(t)=sum(y(t:t+(K-1)));
    X_K(t,:)=sum(X(t:t+(K-1),:));
    Z_tilde_K(t,:)=sum(Z_tilde(t:t+(K-1),:));
end;

% Construct matrices for demeaned variables and instruments; see Sec 5.1

n_K=length(y_K)-1;
y_bar_K=mean(y_K(2:end));
x_bar_K=mean(X_K(1:end-1,:))';
z_tilde_bar_K=mean(Z_tilde_K(1:end-K,:))';
Y_K_under=y_K(2:end)-y_bar_K;
X_K_under=X_K(1:end-1,:)-kron(x_bar_K',ones(n_K,1));
Z_tilde_K=Z_tilde_K(1:end-1,:);
Z_tilde=Z_tilde(1:end-K,:);

% IVX estimation of demeaned predictive regression; see eq (33)

A_tilde_IVX_K=inv(X_K_under'*Z_tilde)*(Z_tilde'*Y_K_under);

% Compute IVX-Wald statistic; see eq (34)

Omega_hat_FM=Sigma_hat_ee-Omega_hat_eu*inv(Omega_hat_uu)*Omega_hat_eu';
M_K=(Z_tilde_K'*Z_tilde_K)*Sigma_hat_ee-...
    n_K*(z_tilde_bar_K*z_tilde_bar_K')*Omega_hat_FM;
Q_H_K=inv(Z_tilde'*X_K_under)*M_K*inv(X_K_under'*Z_tilde);
W_IVX_K=A_tilde_IVX_K'*inv(Q_H_K)*A_tilde_IVX_K;
p_value=1-chi2cdf(W_IVX_K,size(X,2));
