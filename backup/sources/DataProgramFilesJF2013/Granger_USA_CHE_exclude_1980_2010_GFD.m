%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Granger_USA_CHE_exclude_1980_2010_GFD.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 05-25-2012

clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading data, 1980:02-2010:12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Data_1980_2010_equity_premium_GFD;
data_equity_premium=100*data_equity_premium; % expressing in percent
Data_1980_2010_bill;
Data_1980_2010_dividend_yield;
data_dy=log(data_dividend_yield);
Data_1980_2010_equity_premium_GFD_exclude_last_day;
data_equity_premium_exclude=100*data_equity_premium_exclude;

%%%%%%%%%%%%%%%
% Preliminaries
%%%%%%%%%%%%%%%

X_1=data_bill;
X_2=data_dy;
[T,N]=size(data_equity_premium);
j=9; % Switzerland

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constructing data matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

intercept_pool=kron(eye(N),ones(T-1,1));
Y=data_equity_premium(2:T,:);
Y_pool=Y(:);
Y_lag=data_equity_premium(1:T-1,:);
Y_lag_pool=Y_lag(:);
X_1=X_1(1:T-1,:);
X_1_pool=X_1(:);
X_2=X_2(1:T-1,:);
X_2_pool=X_2(:);
USA_lag=zeros(T-1,N);
for i=1:N;
    if i~=N;
        if (i==2);
            USA_lag(:,i)=data_equity_premium(1:T-1,N);
        else
            USA_lag(:,i)=data_equity_premium_exclude(1:T-1,N);
        end;
    end;
end;
USA_lag_pool=USA_lag(:);
j_lag=zeros(T-1,N);
for i=1:N;
    if i~=j;
        if (i==1) || (i==6);
            j_lag(:,i)=data_equity_premium_exclude(1:T-1,j);
        else
            j_lag(:,i)=data_equity_premium(1:T-1,j);
        end;
    end;
end;
j_lag_pool=j_lag(:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computing pooled OLS estimates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X_pool=[USA_lag_pool j_lag_pool Y_lag_pool X_1_pool X_2_pool ...
    intercept_pool];
results_pool=ols(Y_pool,X_pool);
e_hat=zeros(T-1,N);
for i=1:N;
    e_hat(:,i)=results_pool.resid((i-1)*(T-1)+1:i*(T-1));
end;
beta_hat=results_pool.beta(1:2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generating bootstrapped replications
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

B=2000;
beta_hat_star=zeros(B,2);
for b=1:B;
    w_star=randn(T-1,1);
    e_hat_star=kron(ones(1,N),w_star).*e_hat;
    e_hat_pool_star=e_hat_star(:);
    Y_pool_star=X_pool*results_pool.beta+e_hat_pool_star;
    results_pool_star=ols(Y_pool_star,X_pool);
    beta_hat_star(b,:)=results_pool_star.beta(1:2)';
    disp(b);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computing 90% confidence intervals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CI_90=zeros(2,2);
CI_90(:,1)=Compute_bootstrap_CI_90(beta_hat(1),beta_hat_star(:,1),1);
CI_90(:,2)=Compute_bootstrap_CI_90(beta_hat(2),beta_hat_star(:,2),1);
results=[beta_hat' ; CI_90];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Writing results to spreadsheet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%xlswrite('Returns_international_results_1980_2010',results,...
%    'Granger causality--GFD','ci65');
