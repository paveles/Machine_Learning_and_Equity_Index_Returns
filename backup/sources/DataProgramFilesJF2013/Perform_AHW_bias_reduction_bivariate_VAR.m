function [Theta_c_hat,Phi_c_hat,v_c_hat]=Perform_AHW_bias_reduction_bivariate_VAR(x_1,x_2)

% Last modified: 05-15-2012

% Generates Amihud et al. (2009) iterative bias-reduced estimates of the
% bivariate VAR(1) process:
%
% x_1(t) = Theta_1 + Phi_11*x_1(t-1) + Phi_12*x_2(t-1) + v_1(t)
% x_2(t) = Theta_2 + Phi_21*x_1(t-1) + Phi_22*x_2(t-1) + v_2(t)
%
% The function calls the function Perform_YK_bias_reduction_bivariate_VAR.m.
%
% Input
%
% x_1 = (n+1)-vector of x_1(t) observations
% x_2 = (n+1)-vector of x_2(t) observations
%
% Output
%
% Theta_c_hat = 2-vector of bias-reduced intercept estimates
% Phi_c_hat   = 2x2 matrix of bias-reduced slope coefficient estimates
% v_c_hat     = nx2 matrix of bias-reduced residuals
%
% Reference
%
% Y. Amihud, C.M. Hurvich, Y. Wang (2009), "Multiple-Predictor Regressions:
% Hypothesis Testing," Review of Financial Studies 22, 413-434

n=size(x_1,1)-1;
Theta_hat=zeros(2,1);
Phi_hat=zeros(2,2);
results_1=ols(x_1(2:n+1),[ones(n,1) x_1(1:n) x_2(1:n)]);
Theta_hat(1)=results_1.beta(1);
Phi_hat(1,:)=results_1.beta(2:3)';
results_2=ols(x_2(2:n+1),[ones(n,1) x_1(1:n) x_2(1:n)]);
Theta_hat(2)=results_2.beta(1);
Phi_hat(2,:)=results_2.beta(2:3)';
lambda=eig(Phi_hat');
modulus_lambda=abs(sort(lambda,'descend'));
if modulus_lambda(1)>=1;
    x_bar_star=mean([x_1 x_2]);
    x_dev=[x_1(2:n+1) x_2(2:n+1)]-kron(x_bar_star,ones(n,1));
    x_lag_dev=[x_1(1:n) x_2(1:n)]-kron(x_bar_star,ones(n,1));
    Phi_YW=(x_dev'*x_lag_dev)*inv(x_lag_dev'*x_lag_dev);
    Phi_hat=Phi_YW;
    Theta_hat=mean([x_1(2:n+1) x_2(2:n+1)])'-Phi_hat*mean([x_1(1:n) x_2(1:n)])';
end;
v_hat=[x_1(2:n+1) x_2(2:n+1)]-[ones(n,1) x_1(1:n) x_2(1:n)]*[Theta_hat' ; Phi_hat'];
Theta_c_hat=Theta_hat;
Phi_c_hat=Phi_hat;
v_c_hat=v_hat;
for k=1:10;
    [Theta_c_hat_new,Phi_c_hat_new]=Perform_YK_bias_reduction_bivariate_VAR(Theta_c_hat,Phi_c_hat,v_c_hat,n);
    lambda_k=eig(Phi_c_hat_new');
    modulus_lambda_k=abs(sort(lambda_k,'descend'));
    if modulus_lambda_k(1)<1;
        Theta_c_hat=Theta_c_hat_new;
        Phi_c_hat=Phi_c_hat_new;
    end;
    v_c_hat=[x_1(2:n+1) x_2(2:n+1)]-[ones(n,1) x_1(1:n) x_2(1:n)]*[Theta_c_hat' ; Phi_c_hat'];
end;
