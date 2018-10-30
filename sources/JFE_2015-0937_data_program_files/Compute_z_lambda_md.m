function [z_lambda_md]=Compute_z_lambda_md(y,delta)

% Last modified: 08-18-2015

% Computes Harvey, Leybourne, and Taylor (2007) robust trend test
% statistic.
%
% Input
%
% y     = T-vector of dependent variable observations
% delta = parameter for test statistic (use delta = 1 or 2)
%
% Output
%
% z_lambda_md = test statistic
%
% Reference
%
% Harvey, DI, SJ Leybourne, and AMR Taylor (2007), "A Simple, Robust and
% and Powerful Test of the Trend Hypothesis," Journal of Econometrics
% 141, 1302-1330
%
% NB: requires Compute_MZ_ADF_GLS.m

% Estimate I(0) model via OLS

T=length(y);
trend=(1:1:T)';
T_star=T-1;
Z=[ones(T,1) trend];
results_0=ols(y,Z);
beta_hat=results_0.beta(2);
u_hat=results_0.resid;

% Select bandwidth for I(0) model using Newey and West (1994)

n_hat=round(4*(T/100)^(2/9));
gamma_hat_0=T^(-1)*(u_hat'*u_hat);
s_hat_0=gamma_hat_0;
s_hat_2=0;
for i=1:n_hat;
    gamma_hat_i=T^(-1)*(u_hat(i+1:end)'*u_hat(1:end-i));
    s_hat_0=s_hat_0+2*gamma_hat_i;
    s_hat_2=s_hat_2+2*i^2*gamma_hat_i;
end;
gamma_hat=1.3221*((s_hat_2/s_hat_0)^2)^(1/5);
m_hat=min([T ; round(gamma_hat*T^(1/5))]);

% Estimate long-run variance for I(0) using quadratic spectral kernel

omega2_hat_u=gamma_hat_0;
for j=1:T-1;
    z_j=(6*pi/5)*(j/m_hat);
    h_j=(3/z_j^2)*((sin(z_j)/z_j)-cos(z_j));
    gamma_hat_j=T^(-1)*(u_hat(j+1:end)'*u_hat(1:end-j));
    omega2_hat_u=omega2_hat_u+2*h_j*gamma_hat_j;
end;

% Estimate I(1) model via OLS

results_1=ols(y(2:end)-y(1:end-1),ones(T_star,1));
beta_tilde=results_1.beta;
v_tilde=results_1.resid;

% Select bandwidth for I(1) model using Newey and West (1994)

n_tilde=round(4*(T_star/100)^(2/9));
gamma_tilde_0=T_star^(-1)*(v_tilde'*v_tilde);
s_tilde_0=gamma_tilde_0;
s_tilde_2=0;
for i=1:n_tilde;
    gamma_tilde_i=T_star^(-1)*(v_tilde(i+1:end)'*v_tilde(1:end-i));
    s_tilde_0=s_tilde_0+2*gamma_tilde_i;
    s_tilde_2=s_tilde_2+2*i^2*gamma_tilde_i;
end;
gamma_tilde=1.3221*((s_tilde_2/s_tilde_0)^2)^(1/5);
m_tilde=min([T_star ; round(gamma_tilde*T_star^(1/5))]);

% Estimate long-run variance for I(0) using quadratic spectral kernel

omega2_tilde_v=gamma_tilde_0;
for j=1:T-2;
    z_j=(6*pi/5)*(j/m_tilde);
    h_j=(3/z_j^2)*((sin(z_j)/z_j)-cos(z_j));
    gamma_tilde_j=T_star^(-1)*(v_tilde(j+1:end)'*v_tilde(1:end-j));
    omega2_tilde_v=omega2_tilde_v+2*h_j*gamma_tilde_j;
end;

% Compute t-statistics w/adjustment for I(1) model

s_0=sqrt(omega2_hat_u/sum((trend-mean(trend)).^2));
s_1=sqrt(omega2_tilde_v/T_star);
z_0=beta_hat/s_0;
z_1=beta_tilde/s_1;
sigma2_hat_u=(T-2)^(-1)*(u_hat'*u_hat);
R_delta=(omega2_tilde_v/(T^(-1)*sigma2_hat_u))^delta;
if delta==1;
    gamma_xd=[0.04411 0.03952 0.03292];
elseif delta==2;
    gamma_xd=[0.00149 0.00115 0.00071];
end;
z_1_md=gamma_xd*R_delta*z_1;

% Compute unit root and stationarity test statistics

[~,DF_GLS_tau]=Compute_MZ_ADF_GLS(y);
S_hat=cumsum(u_hat);
eta_hat_tau=(S_hat'*S_hat)/(T^2*omega2_hat_u);
lambda=exp(-0.00025*(DF_GLS_tau/eta_hat_tau)^2);

% Compute robust t-statistic for linear trend

z_lambda_md=(1-lambda)*z_0+lambda*z_1_md;
