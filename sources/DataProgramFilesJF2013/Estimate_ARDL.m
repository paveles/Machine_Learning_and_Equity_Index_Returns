function [results_ARDL,beta]=Estimate_ARDL(Y,X,lag_orders)

% Last modified: 05-19-2012

% Estimate ARDL model
%
% Input
%
% Y         = T-vector of dependent variable observations
% X         = T-vector of x variable observations
% lag_oders = 1-by-2 vector of lag orders (q_1,q_2)
%
% Output
%
% results_ARDL = (sum of DL coefficients, t-stat, R-square,
%                F-stat for test of zero DL coefficients, p-value)
% beta         = estimated coefficient vector

T=size(Y,1);
q_1=lag_orders(1);
q_2=lag_orders(2);
q_max=max([q_1 ; q_2]);
Y_0=Y(q_max+1:end);
Y_lags=[];
X_lags=[];
for q=1:q_max;
    Y_lags=[Y_lags Y(q_max-(q-1):T-q)];
    X_lags=[X_lags X(q_max-(q-1):T-q)];
end;
T=size(Y_0,1);
if q_1==0;
    X=[ones(T,1) X_lags];
else
    X=[ones(T,1) Y_lags(:,1:q_1) X_lags(:,1:q_2)];
end;
results_NW=NW_estimation(Y_0,X,0);
beta=results_NW.beta;
cov_beta=results_NW.cov;
R=[zeros(q_2,q_1+1) eye(q_2)];
F_stat=(R*beta)'*inv(R*cov_beta*R')*(R*beta)/q_2;
p_value=1-fdis_cdf(F_stat,q_2,T-size(X,2));
sum_beta=sum(beta(q_1+2:size(X,2)));
R_sum=[zeros(1,q_1+1) ones(1,q_2)];
SE_sum=sqrt(diag(R_sum*cov_beta*R_sum'));
t_stat_sum_beta=sum_beta/SE_sum;
results_ARDL=[sum_beta t_stat_sum_beta results_NW.rsqr F_stat p_value];
