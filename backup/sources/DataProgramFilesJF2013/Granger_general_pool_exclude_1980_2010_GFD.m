%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Granger_general_pool_exclude_1980_2010_GFD.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 05-24-2012

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimating pooled general model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[T,N]=size(data_equity_premium);
intercept_pool=kron(eye(N),ones(T-1,1));
bill_lag=data_bill(1:T-1,:);
bill_lag=bill_lag(:);
dy_lag=data_dy(1:T-1,:);
dy_lag=dy_lag(:);
i_lag=data_equity_premium(1:T-1,:);
i_lag=i_lag(:);
j_lag=zeros(size(i_lag,1),N);
for j=1:N;
    for i=1:N;
        if i~=j;
            if (j==1) || (j==6);
                j_lag((i-1)*(T-1)+1:i*(T-1),j)=...
                    data_equity_premium(1:T-1,j);
            elseif (j==2) || (j==4) || (j==11);
                if (i==2) || (i==11);
                    j_lag((i-1)*(T-1)+1:i*(T-1),j)=...
                        data_equity_premium(1:T-1,j);
                else
                    j_lag((i-1)*(T-1)+1:i*(T-1),j)=...
                        data_equity_premium_exclude(1:T-1,j);
                end;
            else;
                if (i==1) || (i==6);
                    j_lag((i-1)*(T-1)+1:i*(T-1),j)=...
                        data_equity_premium_exclude(1:T-1,j);
                else
                    j_lag((i-1)*(T-1)+1:i*(T-1),j)=...
                        data_equity_premium(1:T-1,j);
                end;
            end;
        end;
    end;
end;
X_pool=[j_lag i_lag bill_lag dy_lag intercept_pool];
Y=data_equity_premium(2:T,:);
Y_pool=Y(:);
results_pool=ols(Y_pool,X_pool);
beta_hat=results_pool.beta;
e_hat=zeros(T-1,N);
for i=1:N;
    e_hat(:,i)=results_pool.resid((i-1)*(T-1)+1:i*(T-1));
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generating bootstrapped confidence intervals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

B=2000;
beta_hat_star=zeros(B,N);
for b=1:B;
    w_star=randn(T-1,1);
    e_hat_star=kron(ones(1,N),w_star).*e_hat;
    e_hat_star=e_hat_star(:);
    Y_pool_star=X_pool*beta_hat+e_hat_star;
    results_pool_star=ols(Y_pool_star,X_pool);
    beta_hat_star(b,:)=results_pool_star.beta(1:N)';
    disp(b);
end;
CI_90=zeros(2,N);
for i=1:N;
    CI_90(:,i)=Compute_bootstrap_CI_90(beta_hat(i),beta_hat_star(:,i),1);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Writing results to spreadsheet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%xlswrite('Returns_international_results_1980_2010',...
%    [beta_hat(1:N)' ; CI_90],'Granger causality--GFD','av65');
