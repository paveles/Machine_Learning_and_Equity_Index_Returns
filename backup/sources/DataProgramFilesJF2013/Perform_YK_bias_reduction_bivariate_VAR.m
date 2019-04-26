function [Theta_c_hat,Phi_c_hat]=Perform_YK_bias_reduction_bivariate_VAR(Theta_hat,Phi_hat,v_hat,n)

% Performs Yamamoto and Kunitomo (1984) bias-reduced estimation of a
% bivariate VAR(1) model. As shown by Engsted and Pedersen (2011), the
% YK bias formula is equivalent to the Nicholls and Pope (1988) formula.
%
% Input
%
% Theta_hat = 2-vector of OLS intercept estimates
% Phi_hat   = 2x2 matrix of OLS slope coefficient estimates
% n         = number of usable observations
%
% Output
%
% Theta_c_hat = 2-vector of bias-reduced intercpt estimates
% Phi_c_hat   = 2x2 matix of bias-reduced slope coefficient estimates
% v_c_hat     = nx2 matrix of bias-reduced residuals
%
% References
%
% Engsted, T. and T.Q. Pedersen (2011), "Bias-Correction in Vector
% Autoregressive Models: A Simulation Study," CREATES Research Paper
% 2011-18
%
% Nicholls, D.F. and A.L. Pope (1988), "Bias in the estimation of
% multivariate autoregressions," Australian Journal of Statistics 30A,
% 296-309
%
% Yamamoto, T. and N, Kunitomo (1984), "Asymptotic bias of the least
% squares estimator for multivariate autoregression models," Annals of
% the Institute of Statistical Mathematics 36, 419-430

Sigma_v_hat=cov(v_hat);
Phi_Sigma_Phi=zeros(2,2);
for j=0:n;
    Phi_Sigma_Phi=Phi_Sigma_Phi+(Phi_hat^j)*Sigma_v_hat*((Phi_hat')^j);
end;
b_YK_Phi=zeros(2,2);
for i=0:n;
    b_YK_Phi=b_YK_Phi+Sigma_v_hat*(((Phi_hat')^j)+((Phi_hat')^j)*trace((Phi_hat)^(i+1))+((Phi_hat')^(2*i+1)))*inv(Phi_Sigma_Phi);
end;
Phi_c_hat=Phi_hat+(1/n)*b_YK_Phi;
b_YK_Theta=-b_YK_Phi*(inv(eye(2)-Phi_hat))*Theta_hat;
Theta_c_hat=Theta_hat+(1/n)*b_YK_Theta;
