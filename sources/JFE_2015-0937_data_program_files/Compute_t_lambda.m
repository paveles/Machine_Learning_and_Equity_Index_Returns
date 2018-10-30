function [t_lambda]=Compute_t_lambda(y)

% Last modified: 08-18-2015

% Computes the Harvey, Leybourne, and Taylor (2009) robust breaking trend
% test statistic for their Model A.
%
% Input
%
% y = T-vector of dependent variable observations
%
% Output
%
% t_lambda = test statistic
%
% Reference
%
% Harvey, DI, SJ Leyboure, and AMR Taylor (2009), "Simple, Robust, and
% Powerful Tests of the Breaking Trend Hypothesis," Econometric Theory
% 25, 995-1029

T=length(y);
trend=(1:1:T)';
l=round(4*(T/100)^(1/4));
tau_L=round(0.10*T);
tau_U=round(0.90*T);
Lambda=(tau_L:1:tau_U)';
t_0=nan(length(Lambda),1);
S_0=nan(length(Lambda),1);
t_1=nan(length(Lambda),1);
S_1=nan(length(Lambda),1);
for t=1:length(Lambda);
    T_star=Lambda(t);
    DT=(trend>T_star).*(trend-T_star);
    X_DT=[ones(T,1) trend DT];
    results_DT=ols(y,X_DT);
    gamma_hat=results_DT.beta(3);
    u_hat=results_DT.resid;
    gamma_hat_0=T^(-1)*(u_hat'*u_hat);
    omega2_hat=gamma_hat_0;
    for j=1:l;
        gamma_hat_j=T^(-1)*(u_hat(j+1:end)'*u_hat(1:end-j));
        omega2_hat=omega2_hat+2*(1-(j/(l+1)))*gamma_hat_j;
    end;
    Cov_DT=omega2_hat*inv(X_DT'*X_DT);
    t_0(t)=gamma_hat/sqrt(Cov_DT(3,3));
    cumulative_sum_0=cumsum(u_hat);
    S_0(t)=(T^2*omega2_hat)^(-1)*(cumulative_sum_0'*cumulative_sum_0);
    DU=trend>T_star;
    dy=y(2:end)-y(1:end-1);
    X_DU=[ones(T-1,1) DU(2:end)];
    results_DU=ols(dy,X_DU);
    gamma_tilde=results_DU.beta(2);
    v_tilde=results_DU.resid;
    gamma_tilde_0=T^(-1)*(v_tilde'*v_tilde);
    omega2_tilde=gamma_tilde_0;
    for j=1:l;
        gamma_tilde_j=(T-1)^(-1)*(v_tilde(j+1:end)'*v_tilde(1:end-j));
        omega2_tilde=omega2_tilde+2*(1-(j/(l+1)))*gamma_tilde_j;
    end;
    Cov_DU=omega2_tilde*inv(X_DU'*X_DU);
    t_1(t)=gamma_tilde/sqrt(Cov_DU(2,2));
    cumulative_sum_1=cumsum(v_tilde);
    S_1(t)=((T-1)^2*omega2_tilde)^(-1)*(cumulative_sum_1'*cumulative_sum_1);
end;
[t_0_star,t_0_star_index]=max(abs(t_0));
[t_1_star,t_1_star_index]=max(abs(t_1));
S_0=S_0(t_0_star_index);
S_1=S_1(t_1_star_index);
g=500;
lambda=exp(-(g*S_0*S_1)^2);
m=[0.835 0.853 0.890];
t_lambda=lambda*t_0_star*ones(1,3)+m*(1-lambda)*t_1_star;
