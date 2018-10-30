%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Granger_AdENET_FRA_exclude_1980_2010_GFD.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fitting adaptive elastic net model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[T,N]=size(data_equity_premium);
nfolds=5;
i=3; % France
Y_i=data_equity_premium(2:T,i);
X_i=[data_equity_premium(1:T-1,:) data_bill(1:T-1,i) data_dy(1:T-1,i)];
X_i(:,[2 4 11])=data_equity_premium_exclude(1:T-1,[2 4 11]);
[select_i,beta_i,tuning_i]=Perform_selection_AdENET(Y_i,X_i,nfolds,0);
beta_i=beta_i(1:N)';
beta_i(i)=0;
tuning_i=tuning_i';
disp(i);
disp(tuning_i);
disp(beta_i);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generating bootstrapped confidence intervals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CI_90_i=zeros(2,N);
B=2000;
results_ols_i=ols(zscore(Y_i),zscore(X_i));
w_hat_i=(abs(results_ols_i.beta)).^(-tuning_i(1));
opts.alpha=tuning_i(2);
opts.lambda=tuning_i(3);
opts.exclude=[];
opts.penalty_factor=w_hat_i;
options=glmnetSet(opts);
fit_hat_i=glmnet(X_i,Y_i,'gaussian',options);
exclude_i=fit_hat_i.beta==0;
exclude_i=find(exclude_i);
exclude_i=sort(exclude_i);
intercept_i=mean(Y_i)-mean(X_i)*fit_hat_i.beta;
e_hat_i=Y_i-(intercept_i+X_i*fit_hat_i.beta);
beta_hat_i_star=zeros(B,size(X_i,2));
draw_1=randn(T-1,B);
for b=1:B;
    Y_i_star=intercept_i+X_i*fit_hat_i.beta+e_hat_i.*draw_1(:,b);
    opts.alpha=tuning_i(2);
    opts.lambda=tuning_i(3);
    opts.exclude=exclude_i;
    opts.penalty_factor=w_hat_i;
    options=glmnetSet(opts);
    fit_hat_i_star=glmnet(X_i,Y_i_star,'gaussian',options);
    beta_hat_i_star(b,:)=fit_hat_i_star.beta';
end;
for j=1:N;
    CI_90_i(:,j)=Compute_bootstrap_CI_90(fit_hat_i.beta(j),...
        beta_hat_i_star(:,j),1);
end;
CI_90_i(:,i)=zeros(2,1);
disp(CI_90_i);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Writing results to spreadsheet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%xlswrite('Returns_international_results_1980_2010',...
%    [beta_i tuning_i],'Granger causality--GFD','av14');
%xlswrite('Returns_international_results_1980_2010',...
%    CI_90_i,'Granger causality--GFD','av15');
