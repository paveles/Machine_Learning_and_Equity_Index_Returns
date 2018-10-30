%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Granger_pairwise_raw_1980_2010_GFD.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 05-20-2012

clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading data, 1980:02-2010:12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Data_1980_2010_equity_premium_GFD;
data_equity_premium=100*data_equity_premium; % expressing in percent
Data_1980_2010_bill;
Data_1980_2010_dividend_yield;
data_dy=log(data_dividend_yield);

%%%%%%%%%%%%%%%
% Preliminaries
%%%%%%%%%%%%%%%

X_1=data_bill;
X_2=data_dy;
[T,N]=size(data_equity_premium);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Testing predictive power of lagged country returns
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

results_all=zeros(4,N,N+1);
for j=1:N;
    [results_j,R_squared_j]=...
        Estimate_Granger_pairwise_GMM(data_equity_premium,...
        data_equity_premium,X_1,X_2,j);
    for i=1:(N-1);
        if j==1;
            results_all(1:2,j,i+1)=results_j(i,:)';
            results_all(4,j,i+1)=100*R_squared_j(i);
        else
            if i<j;
                results_all(1:2,j,i)=results_j(i,:)';
                results_all(4,j,i)=100*R_squared_j(i);
            else
                results_all(1:2,j,i+1)=results_j(i,:)';
                results_all(4,j,i+1)=100*R_squared_j(i);
            end;
        end;
    end;
    results_all(1:2,j,N+1)=results_j(end,:)';
    results_all(4,j,N+1)=100*R_squared_j(end);
    disp(j);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimating DGP parameters for wild bootstrap
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

beta=zeros(4,N);
e=zeros(T-1,N);
Theta=zeros(2,N);
Phi=zeros(2,2,N);
v=zeros(T-1,2,N);
for i=1:N;
    results_0=ols(data_equity_premium(2:T,i),...
        [data_equity_premium(1:T-1,i) X_1(1:T-1,i) X_2(1:T-1,i) ...
        ones(T-1,1)]);
    beta(:,i)=results_0.beta;
    e(:,i)=results_0.resid;
    [Theta(:,i),Phi(:,:,i),v(:,:,i)]=...
        Perform_AHW_bias_reduction_bivariate_VAR(X_1(:,i),X_2(:,i));
    disp(i);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generating pseudo data using wild boostrap under null
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

B=2000;
Y_star=zeros(T,N,B);
X_1_star=zeros(T,N,B);
X_2_star=zeros(T,N,B);
for b=1:B;
    Y_star(1,:,b)=data_equity_premium(1,:);
    X_1_star(1,:,b)=X_1(1,:);
    X_2_star(1,:,b)=X_2(1,:);
    w=randn(T-1,1);
    for t=2:T;
        for i=1:N;
            Y_star(t,i,b)=[Y_star(t-1,i,b) X_1_star(t-1,i,b) ...
                X_2_star(t-1,i,b) 1]*beta(:,i)+w(t-1)*e(t-1,i);
            X_1_star(t,i,b)=Theta(1,i)+Phi(1,:,i)*[X_1_star(t-1,i,b) ; ...
                X_2_star(t-1,i,b)]+w(t-1)*v(t-1,1,i);
            X_2_star(t,i,b)=Theta(2,i)+Phi(2,:,i)*[X_1_star(t-1,i,b) ; ...
                X_2_star(t-1,i,b)]+w(t-1)*v(t-1,2,i);
        end;
    end;
    disp(b);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computing statistics for wild bootstrapped pseudo samples
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

stats_boot=zeros(N+1,N,B);
for b=1:B;
    for j=1:N;
        [results_j_star,R_squared_j_star]=...
            Estimate_Granger_pairwise_GMM(Y_star(:,:,b),...
            Y_star(:,:,b),X_1_star(:,:,b),X_2_star(:,:,b),j);
        for i=1:(N-1);
            if j==1;
                stats_boot(i+1,j,b)=results_j_star(i,2);
            else
                if i<j;
                    stats_boot(i,j,b)=results_j_star(i,2);
                else
                    stats_boot(i+1,j,b)=results_j_star(i,2);
                end;
            end;
        end;
        stats_boot(N+1,j,b)=results_j_star(end,2);
        disp([b j]);
    end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computing wild bootstrapped p-values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for j=1:N;
    for i=1:N+1;
        stats_boot_i_j=stats_boot(i,j,:);
        stats_p_i_j=stats_boot_i_j>results_all(2,j,i);
        results_all(3,j,i)=sum(stats_p_i_j)/B;
    end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Writing results to spreadsheet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%xlswrite('Returns_international_results_1980_2010',results_all(:,:,1),...
%    'Granger causality--GFD','b4');
%xlswrite('Returns_international_results_1980_2010',results_all(:,:,2),...
%    'Granger causality--GFD','b9');
%xlswrite('Returns_international_results_1980_2010',results_all(:,:,3),...
%    'Granger causality--GFD','b14');
%xlswrite('Returns_international_results_1980_2010',results_all(:,:,4),...
%    'Granger causality--GFD','b19');
%xlswrite('Returns_international_results_1980_2010',results_all(:,:,5),...
%    'Granger causality--GFD','b24');
%xlswrite('Returns_international_results_1980_2010',results_all(:,:,6),...
%    'Granger causality--GFD','b29');
%xlswrite('Returns_international_results_1980_2010',results_all(:,:,7),...
%    'Granger causality--GFD','b34');
%xlswrite('Returns_international_results_1980_2010',results_all(:,:,8),...
%    'Granger causality--GFD','b39');
%xlswrite('Returns_international_results_1980_2010',results_all(:,:,9),...
%    'Granger causality--GFD','b44');
%xlswrite('Returns_international_results_1980_2010',results_all(:,:,10),...
%    'Granger causality--GFD','b49');
%xlswrite('Returns_international_results_1980_2010',results_all(:,:,11),...
%    'Granger causality--GFD','b54');
%xlswrite('Returns_international_results_1980_2010',results_all(:,:,12),...
%    'Granger causality--GFD','b65');
