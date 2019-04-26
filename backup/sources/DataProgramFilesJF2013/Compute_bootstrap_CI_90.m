function [CI_90]=Compute_bootstap_CI_90(beta_hat,beta_hat_star,select)

% Last modified: 05-24-2012

% Computes bias-corrected bootstrap 90% confidence intervals.
%
% Input
%
% beta_hat      = scalar parameter estimate based on original data
% beta_hat_star = B-vector of bootstrap replications
% select        = 0/1 value to indicate method (= 0 for bias-corrected
%                 percentile method, = 1 for bias-corrected bootstrap
%                 method)
%
% Output
%
% CI_90 = 2-vector of 90% confidence inteval bounds

if select==0;
    z_alpha=-1.645;
    z_1_alpha=1.645;
    indicator=beta_hat_star<beta_hat;
    z_cdf=mean(indicator);
    z_hat_0=norm_inv(z_cdf);
    alpha_1=norm_cdf(2*z_hat_0+z_alpha);
    alpha_2=norm_cdf(2*z_hat_0+z_1_alpha);
    CI_90=prctile(beta_hat_star,100*[alpha_1 alpha_2]);
elseif select==1;
    B=size(beta_hat_star,1);
    beta_bar_star=mean(beta_hat_star);
    s_beta_star=((1/(B-1))*...
         sum((beta_hat_star-beta_bar_star*ones(B,1)).^2))^(0.5);
    CI_90=[2*beta_hat-beta_bar_star-s_beta_star*1.645 ...
        2*beta_hat-beta_bar_star+s_beta_star*1.645];
end;
CI_90=CI_90';
