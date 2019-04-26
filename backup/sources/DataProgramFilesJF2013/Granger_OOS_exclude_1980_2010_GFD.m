%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Granger_OOS_exclude_1980_2010_GFD.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preliminaries for out-of-sample forecasts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[T,N]=size(data_equity_premium);
R=(1984-1979)*12-1; % 1980:02-1984:12 in-sample period
P=T-R;
FC_baseline=zeros(P,N-1,3);
FC_baseline_pool=zeros(P,N-1,3);
FC_USA=zeros(P,N-1,3);
FC_USA_pool=zeros(P,N-1,3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generating out-of-sample forecasts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for t=1:P;
    for i=1:N-1;
        Y_i=data_equity_premium(2:R+(t-1),i);
        FC_baseline(t,i,1)=mean(data_equity_premium(1:R+(t-1),i));
        FC_baseline_pool(t,i,1)=FC_baseline(t,i,1);
        X_i_AR=[ones(size(Y_i,1),1) data_equity_premium(1:R+(t-2),i)];
        results_i_AR=ols(Y_i,X_i_AR);
        FC_baseline(t,i,2)=[1 data_equity_premium(R+(t-1),i)]*...
            results_i_AR.beta;
        X_i_PR=[ones(size(Y_i,1),1) data_bill(1:R+(t-2),i) ...
            data_dy(1:R+(t-2),i)];
        results_i_PR=ols(Y_i,X_i_PR);
        FC_baseline(t,i,3)=[1 data_bill(R+(t-1),i) ...
            data_dy(R+(t-1),i)]*results_i_PR.beta;
        if i==2;
            X_i_HA_USA=[ones(size(Y_i,1),1) ...
                data_equity_premium(1:R+(t-2),N)];
            results_i_HA_USA=ols(Y_i,X_i_HA_USA);
            FC_USA(t,i,1)=[1 data_equity_premium(R+(t-1),N)]*...
                results_i_HA_USA.beta;
            X_i_AR_USA=[ones(size(Y_i,1),1) ...
                data_equity_premium(1:R+(t-2),i) ...
                data_equity_premium(1:R+(t-2),N)];
            results_i_AR_USA=ols(Y_i,X_i_AR_USA);
            FC_USA(t,i,2)=[1 data_equity_premium(R+(t-1),i) ...
                data_equity_premium(R+(t-1),N)]*results_i_AR_USA.beta;
            X_i_PR_USA=[ones(size(Y_i,1),1) ...
                data_bill(1:R+(t-2),i) data_dy(1:R+(t-2),i) ...
                data_equity_premium(1:R+(t-2),N)];
            results_i_PR_USA=ols(Y_i,X_i_PR_USA);
            FC_USA(t,i,3)=[1 data_bill(R+(t-1),i) data_dy(R+(t-1),i) ...
                data_equity_premium(R+(t-1),N)]*results_i_PR_USA.beta;
        else
            X_i_HA_USA=[ones(size(Y_i,1),1) ...
                data_equity_premium_exclude(1:R+(t-2),N)];
            results_i_HA_USA=ols(Y_i,X_i_HA_USA);
            FC_USA(t,i,1)=[1 data_equity_premium_exclude(R+(t-1),N)]*...
                results_i_HA_USA.beta;
            X_i_AR_USA=[ones(size(Y_i,1),1) ...
                data_equity_premium(1:R+(t-2),i) ...
                data_equity_premium_exclude(1:R+(t-2),N)];
            results_i_AR_USA=ols(Y_i,X_i_AR_USA);
            FC_USA(t,i,2)=[1 data_equity_premium(R+(t-1),i) ...
                data_equity_premium_exclude(R+(t-1),N)]*...
                results_i_AR_USA.beta;
            X_i_PR_USA=[ones(size(Y_i,1),1) ...
                data_bill(1:R+(t-2),i) data_dy(1:R+(t-2),i) ...
                data_equity_premium_exclude(1:R+(t-2),N)];
            results_i_PR_USA=ols(Y_i,X_i_PR_USA);
            FC_USA(t,i,3)=[1 data_bill(R+(t-1),i) data_dy(R+(t-1),i) ...
                data_equity_premium_exclude(R+(t-1),N)]*...
                results_i_PR_USA.beta;
        end;
        Y_pool=data_equity_premium(2:R+(t-1),1:N-1);
        Y_pool=Y_pool(:);
        intercept_pool=kron(eye(N-1),ones(R+(t-2),1));
        X_pool_HA_USA=[intercept_pool ...
            kron(ones(N-1,1),data_equity_premium_exclude(1:R+(t-2),N))];
        results_pool_HA_USA=ols(Y_pool,X_pool_HA_USA);
        FC_USA_pool(t,:,1)=([eye(N-1) ...
            kron(ones(N-1,1),data_equity_premium_exclude(R+(t-1),N))]*...
            results_pool_HA_USA.beta)';
        AR_pool=data_equity_premium(1:R+(t-2),1:N-1);
        AR_pool=AR_pool(:);
        X_pool_AR=[intercept_pool AR_pool];
        results_pool_AR=ols(Y_pool,X_pool_AR);
        FC_baseline_pool(t,:,2)=([eye(N-1) ...
            data_equity_premium(R+(t-1),1:N-1)']*results_pool_AR.beta)';
        X_pool_AR_USA=[intercept_pool AR_pool ...
            kron(ones(N-1,1),data_equity_premium_exclude(1:R+(t-2),N))];
        results_pool_AR_USA=ols(Y_pool,X_pool_AR_USA);
        FC_USA_pool(t,:,2)=([eye(N-1) ...
            data_equity_premium(R+(t-1),1:N-1)' ...
            kron(ones(N-1,1),data_equity_premium_exclude(R+(t-1),N))]*...
            results_pool_AR_USA.beta)';
        bill_pool=data_bill(1:R+(t-2),1:N-1);
        bill_pool=bill_pool(:);
        dy_pool=data_dy(1:R+(t-2),1:N-1);
        dy_pool=dy_pool(:);
        X_pool_PR=[intercept_pool bill_pool dy_pool];
        results_pool_PR=ols(Y_pool,X_pool_PR);
        FC_baseline_pool(t,:,2)=([eye(N-1) ...
            data_bill(R+(t-1),1:N-1)' data_dy(R+(t-1),1:N-1)']*...
            results_pool_PR.beta)';
        X_pool_PR_USA=[intercept_pool bill_pool dy_pool ...
            kron(ones(N-1,1),data_equity_premium_exclude(1:R+(t-2),N))];
        results_pool_PR_USA=ols(Y_pool,X_pool_PR_USA);
        FC_USA_pool(t,:,3)=([eye(N-1) ...
            data_bill(R+(t-1),1:N-1)' data_dy(R+(t-1),1:N-1)' ...
            kron(ones(N-1,1),data_equity_premium_exclude(R+(t-1),N))]*...
            results_pool_PR_USA.beta)';
    end;
    disp(t);
end;
actual=data_equity_premium(R+1:T,1:N-1);

%%%%%%%%%%%%%%%%%%%%%%
% Evaluating forecasts
%%%%%%%%%%%%%%%%%%%%%%

e_baseline=zeros(P,N-1,size(FC_baseline,3));
e_baseline_pool=zeros(P,N-1,size(FC_baseline_pool,3));
e_USA=zeros(P,N-1,size(FC_USA,3));
e_USA_pool=zeros(P,N-1,size(FC_USA_pool,3));
MSFE_ratio=zeros(N-1,size(FC_baseline,3));
MSFE_ratio_pool=zeros(N-1,size(FC_baseline_pool,3));
CSFE_difference=zeros(P,N-1,size(FC_USA,3));
CSFE_difference_pool=zeros(P,N-1,size(FC_USA_pool,3));
MSFE_adj=zeros(N-1,size(FC_USA,3),2);
MSFE_adj_pool=zeros(N-1,size(FC_USA_pool,3),2);
for k=1:size(FC_baseline,3);
    e_baseline(:,:,k)=actual-FC_baseline(:,:,k);
    e_baseline_pool(:,:,k)=actual-FC_baseline_pool(:,:,k);
    e_USA(:,:,k)=actual-FC_USA(:,:,k);
    e_USA_pool(:,:,k)=actual-FC_USA_pool(:,:,k);
    MSFE_baseline_k=mean(e_baseline(:,:,k).^2)';
    MSFE_baseline_pool_k=mean(e_baseline_pool(:,:,k).^2)';
    MSFE_USA_k=mean(e_USA(:,:,k).^2)';
    MSFE_USA_pool_k=mean(e_USA_pool(:,:,k).^2)';
    MSFE_ratio(:,k)=MSFE_USA_k./MSFE_baseline_k;
    MSFE_ratio_pool(:,k)=MSFE_USA_pool_k./MSFE_baseline_k;
    SFE_difference_k=e_baseline(:,:,k).^2-e_USA(:,:,k).^2;
    CSFE_difference(:,:,k)=cumsum(SFE_difference_k);
    SFE_difference_pool_k=e_baseline_pool(:,:,k).^2-e_USA_pool(:,:,k).^2;
    CSFE_difference_pool(:,:,k)=cumsum(SFE_difference_pool_k);
    for i=1:N-1;
        [stats_i_k,p_value_i_k]=Perform_CW_test(actual(:,i),...
            FC_baseline(:,i,k),FC_USA(:,i,k));
        MSFE_adj(i,k,1)=stats_i_k(2);
        MSFE_adj(i,k,2)=p_value_i_k(2);
        [stats_pool_i_k,p_value_pool_i_k]=Perform_CW_test(actual(:,i),...
            FC_baseline_pool(:,i,k),FC_USA_pool(:,i,k));
        MSFE_adj_pool(i,k,1)=stats_pool_i_k(2);
        MSFE_adj_pool(i,k,2)=p_value_pool_i_k(2);
        disp([k i]);
    end;
end;
R2OS=100*(1-MSFE_ratio);
R2OS_pool=100*(1-MSFE_ratio_pool);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Collecting results and writing to spreadsheet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

results_all=zeros(3,6,N-1);
for i=1:N-1;
    results_all(:,:,i)=[R2OS(i,:) R2OS_pool(i,:) ; MSFE_adj(i,:,1) ...
        MSFE_adj_pool(i,:,1) ; MSFE_adj(i,:,2) MSFE_adj_pool(i,:,2)];
    disp(i);
end;
xlswrite('Returns_international_results_1980_2010',results_all(:,:,1),...
    'Out-of-sample R-squared','b3');
xlswrite('Returns_international_results_1980_2010',results_all(:,:,2),...
    'Out-of-sample R-squared','b7');
xlswrite('Returns_international_results_1980_2010',results_all(:,:,3),...
    'Out-of-sample R-squared','b11');
xlswrite('Returns_international_results_1980_2010',results_all(:,:,4),...
    'Out-of-sample R-squared','b15');
xlswrite('Returns_international_results_1980_2010',results_all(:,:,5),...
    'Out-of-sample R-squared','b19');
xlswrite('Returns_international_results_1980_2010',results_all(:,:,6),...
    'Out-of-sample R-squared','b23');
xlswrite('Returns_international_results_1980_2010',results_all(:,:,7),...
    'Out-of-sample R-squared','b27');
xlswrite('Returns_international_results_1980_2010',results_all(:,:,8),...
    'Out-of-sample R-squared','b31');
xlswrite('Returns_international_results_1980_2010',results_all(:,:,9),...
    'Out-of-sample R-squared','b35');
xlswrite('Returns_international_results_1980_2010',results_all(:,:,10),...
    'Out-of-sample R-squared','b39');
%save('Granger_OOS_exclude_1980_2010_GFD_store','actual','FC_baseline',...
%    'FC_baseline_pool','FC_USA','FC_USA_pool','CSFE_difference',...
%    'CSFE_difference_pool');
