function [beta_c_hat,SE_beta_c_hat,tstat_beta_c_hat,phi_c_hat,tstat_phi_c_hat,VAR_c_hat]=Perform_mARM_2_predictors(y,x_1,x_2)

% Last modified: 05-16-2012

% This function uses mARM (Amihud et al., 2009) to estimate the slope coefficients
% of a predictive regression model with two predictors. The function calls the
% function Perform_AHW_bias_reduction_bivariate_VAR.m.
%
% Input
%
% y   = (n+1)-vector of dependend variable observations
% x_1 = (n+1)-vector of observations for first predictor
% x_2 = (n+1)-vector of observations for second predictor
%
% Output
%
% beta_c_hat       = 2-vector of bias-reduced slope coefficient estimates
% SE_beta_c_hat    = 2-vector of standard errors
% tstat_beta_c_hat = 2-vector of t-statistics
%
% Reference
%
% Y. Amihud, C.M. Hurvich, Y. Wang (2009), "Multiple-Predictor Regressions:
% Hypothesis Testing," Review of Financial Studies 22, 413-434

[Theta_c_hat,Phi_c_hat,v_c_hat]=Perform_AHW_bias_reduction_bivariate_VAR(x_1,x_2);
n=size(x_1,1)-1;
X=[ones(n,1) x_1(1:n) x_2(1:n)];
X_c=[X v_c_hat];
results_c=ols(y(2:n+1),X_c);
beta_c_hat=results_c.beta(2:3);
phi_c_hat=results_c.beta(4:5);
tstat_phi_c_hat=results_c.tstat(4:5);
Sigma_v_c_hat=(1/n)*(v_c_hat'*v_c_hat);
cov_Phi_c_hat=kron(Sigma_v_c_hat,inv([x_1(1:n) x_2(1:n)]'*[x_1(1:n) x_2(1:n)]));
var_Phi_c_11_hat=cov_Phi_c_hat(1,1);
var_Phi_c_21_hat=cov_Phi_c_hat(3,3);
cov_Phi_c_11_21_hat=cov_Phi_c_hat(1,3);
cov_Phi_c_21_11_hat=cov_Phi_c_hat(3,1);
var_Phi_c_12_hat=cov_Phi_c_hat(2,2);
var_Phi_c_22_hat=cov_Phi_c_hat(4,4);
cov_Phi_c_12_22_hat=cov_Phi_c_hat(2,4);
cov_Phi_c_22_12_hat=cov_Phi_c_hat(4,2);
var_beta_1_c_hat=phi_c_hat(1)^2*var_Phi_c_11_hat+phi_c_hat(2)^2*var_Phi_c_21_hat+...
    2*phi_c_hat(1)*phi_c_hat(2)*cov_Phi_c_11_21_hat+2*phi_c_hat(2)*phi_c_hat(1)*cov_Phi_c_21_11_hat+...
    results_c.bstd(2)^2;
var_beta_2_c_hat=phi_c_hat(1)^2*var_Phi_c_12_hat+phi_c_hat(2)^2*var_Phi_c_22_hat+...
    2*phi_c_hat(1)*phi_c_hat(2)*cov_Phi_c_12_22_hat+2*phi_c_hat(2)*phi_c_hat(1)*cov_Phi_c_22_12_hat+...
    results_c.bstd(3)^2;
SE_beta_c_hat=sqrt([var_beta_1_c_hat ; var_beta_2_c_hat]);
tstat_beta_c_hat=beta_c_hat./SE_beta_c_hat;
VAR_c_hat=[Theta_c_hat Phi_c_hat];