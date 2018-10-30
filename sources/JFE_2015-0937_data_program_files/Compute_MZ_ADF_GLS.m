function [MZ_GLS,ADF_GLS]=Compute_MZ_ADF_GLS(y)

% Last modified: 08-18-2015

% Computes Ng and Perron (2001) unit root statistics to test the null of a
% unit root against the alternative of stationary around a linear trend.
%
% Input
%
% y = T-vector of dependent variable observations
%
% Output
%
% MZ_GLS  = MZ-GLS statistic
% ADF_GLS = ADF-GLS statistic
%
% Reference
%
% Ng, S and P Perron (2001), "Lag Length Selection and the Construction of
% Unit Root Tests with Good Size and Power," Econometrica 69, 1519-1554

T=length(y);
Z=[ones(T,1) (1:1:T)'];
k_max=round(12*(T/100)^(1/4));
results_OLS=ols(y,Z);
y_hat=results_OLS.resid;
d_y_hat=y_hat(2:end)-y_hat(1:end-1);
d_y_hat=[nan(1,1) ; d_y_hat];
MAIC=[(0:1:k_max)' nan(k_max+1,1)];
d_y_hat_lag=nan(T-1-k_max,k_max);
for k=0:k_max;
    if k==0;
        results_k=ols(d_y_hat(k_max+2:end),y_hat(k_max+1:end-1));
    else
        d_y_hat_lag(:,k)=d_y_hat(k_max+1-(k-1):end-k);
        results_k=ols(d_y_hat(k_max+2:end),...
            [y_hat(k_max+1:end-1) d_y_hat_lag(:,1:k)]);
    end;
    b_u_0_k=results_k.beta(1);
    e_u_k=results_k.resid;
    sigma2_u_k=(T-1-k_max)^(-1)*(e_u_k'*e_u_k);
    tau_T_k=sigma2_u_k^(-1)*b_u_0_k^2*...
        (y_hat(k_max+1:end-1)'*y_hat(k_max+1:end-1));
    MAIC(k+1,2)=log(sigma2_u_k)+2*(tau_T_k+k)/(T-1-k_max);
end;
[~,k_min_index]=min(MAIC(:,2));
k=MAIC(k_min_index,1);
c_bar=-13.5;
y_GLS=[y(1) ; y(2:end)-c_bar*y(1:end-1)];
Z_GLS=[Z(1,:) ; Z(2:end,:)-c_bar*Z(1:end-1,:)];
results_GLS=ols(y_GLS,Z_GLS);
y_tilde=results_GLS.resid;
d_y_tilde=y_tilde(2:end)-y_tilde(1:end-1);
d_y_tilde=[nan(1,1) ; d_y_tilde];
if k==0;
    results=ols(d_y_tilde(2:end),y_tilde(1:end-1));
else
    d_y_tilde_lag=nan(T-1-k,k);
    for j=1:k;
        d_y_tilde_lag(:,j)=d_y_tilde(k+1-(j-1):end-j);
    end;
    results=ols(d_y_tilde(k+2:end),[y_tilde(k+1:end-1) d_y_tilde_lag]);
end;
ADF_GLS=results.tstat(1);
b_hat_1=sum(results.beta(2:end));
e_hat=results.resid;
sigma2_hat_k=(T-1-k)^(-1)*(e_hat'*e_hat);
s2_AR=sigma2_hat_k*(1-b_hat_1)^(-2);
MZ_GLS=((T-1)^(-1)*y_tilde(end)^2-s2_AR)*...
    (2*T^(-2)*(y_tilde(1:end-1)'*y_tilde(1:end-1)))^(-1);
