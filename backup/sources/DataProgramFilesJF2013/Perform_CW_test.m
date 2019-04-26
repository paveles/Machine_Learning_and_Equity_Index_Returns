function [stats,p_values]=Perform_CW_test(actual,forecast_1,forecast_2)

% Last modified: 05-25-2012

% Performs the Diebold and Mariano (1995) and Clark and West (2007) tests to
% compare competing forecasts
%
% Input:
%
% actual     = n-vector of actual values
% forecast_1 = n-vector of benchmark forecasts
% forecast_2 = n-vector of competitor forecasts
%
% Output:
%
% stats    = 2-vector: DM stat, MSPE-adjusted stat
% p_values = 2-vector of p-values
%
% References
%
% F.X. Diebold and R.S. Mariano (1995), "Comparing Predictive Accuracy,"
% Journal of Business and Economic Statistics, 13(3), 253-263
%
% T.E. Clark and K.D. West (2007), "Approximately Normal Tests for Equal
% Predictive Accuracy in Nested Models," Journal of Econometrics, 138(1),
% 291-311

e_1=actual-forecast_1;
e_2=actual-forecast_2;
d_hat=e_1.^2-e_2.^2;
f_hat=e_1.^2-(e_2.^2-(forecast_1-forecast_2).^2);
Y_d=d_hat;
Y_f=f_hat;
X=ones(size(d_hat,1),1);
beta_d=((X'*X)^(-1))*(X'*Y_d);
beta_f=((X'*X)^(-1))*(X'*Y_f);
u_d=Y_d-X*beta_d;
u_f=Y_f-X*beta_f;
sig2_d=(u_d'*u_d)/(size(Y_d,1)-1);
sig2_f=(u_f'*u_f)/(size(Y_f,1)-1);
cov_beta_d=sig2_d*((X'*X)^(-1));
cov_beta_f=sig2_f*((X'*X)^(-1));
DM=beta_d/sqrt(cov_beta_d);
MSPE_adjusted=beta_f/sqrt(cov_beta_f);
p_value_d=1-normcdf(DM,0,1);
p_value_f=1-normcdf(MSPE_adjusted,0,1);
stats=[DM ; MSPE_adjusted];
p_values=[p_value_d ; p_value_f];
