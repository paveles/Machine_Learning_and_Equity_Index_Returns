%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Granger_breaks_exclude_1980_2010_GFD.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%
% Preliminaries
%%%%%%%%%%%%%%%

Y=data_equity_premium;
X_1=data_bill;
X_2=data_dy;
[T,N]=size(Y);
results_trend=zeros(3,N-1);
linear_trend=(1:1:T-1)';
qLL=zeros(N-1,1);

%%%%%%%%%%%%%%%%%%%%
% Linear trend model
%%%%%%%%%%%%%%%%%%%%

for i=1:N-1;
    Y_i=data_equity_premium(2:T,i);
    if (i==2);
        X_i=[data_equity_premium(1:T-1,end).*...
            linear_trend data_equity_premium(1:T-1,end) ...
            data_equity_premium(1:T-1,i) X_1(1:T-1,i) X_2(1:T-1,i) ...
            ones(T-1,1)];
    else
        X_i=[data_equity_premium_exclude(1:T-1,end).* ...
            linear_trend data_equity_premium_exclude(1:T-1,end) ...
            data_equity_premium(1:T-1,i) X_1(1:T-1,i) X_2(1:T-1,i) ...
            ones(T-1,1)];        
    end;
    results_i=nwest(Y_i,X_i,0);
    p_tstat_i=norm_cdf(results_i.tstat(1),0,1);
    results_trend(:,i)=[results_i.beta(1) ; results_i.tstat(1) ; ...
        p_tstat_i];
    disp(i);
end;
%xlswrite('Returns_international_results_1980_2010',results_trend(:,1),...
%    'Granger causality--GFD','an4');
%xlswrite('Returns_international_results_1980_2010',results_trend(:,2),...
%    'Granger causality--GFD','an9');
%xlswrite('Returns_international_results_1980_2010',results_trend(:,3),...
%    'Granger causality--GFD','an14');
%xlswrite('Returns_international_results_1980_2010',results_trend(:,4),...
%    'Granger causality--GFD','an19');
%xlswrite('Returns_international_results_1980_2010',results_trend(:,5),...
%    'Granger causality--GFD','an24');
%xlswrite('Returns_international_results_1980_2010',results_trend(:,6),...
%    'Granger causality--GFD','an29');
%xlswrite('Returns_international_results_1980_2010',results_trend(:,7),...
%    'Granger causality--GFD','an34');
%xlswrite('Returns_international_results_1980_2010',results_trend(:,8),...
%    'Granger causality--GFD','an39');
%xlswrite('Returns_international_results_1980_2010',results_trend(:,9),...
%    'Granger causality--GFD','an44');
%xlswrite('Returns_international_results_1980_2010',results_trend(:,10),...
%    'Granger causality--GFD','an49');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Elliott and Muller (2006) test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:N-1;
    Y_i=data_equity_premium(2:T,i);
    Z_i=[data_equity_premium(1:T-1,i) X_1(1:T-1,i) X_2(1:T-1,i) ...
        ones(T-1,1)];
    if (i==2);
        X_i=data_equity_premium(1:T-1,end);
    else
        X_i=data_equity_premium_exclude(1:T-1,end);
    end;
    results_i=ols(Y_i,[Z_i X_i]);
    X_i_e_i=X_i.*results_i.resid;
    V_X_hat=(1/size(X_i_e_i,1))*(X_i_e_i'*X_i_e_i);
    U_hat=(V_X_hat^(-0.5))*X_i_e_i;
    r_bar=1-(10/size(X_i_e_i,1));
    w_hat=zeros(size(X_i_e_i,1),1);
    for t=1:size(X_i_e_i,1);
        if t==1;
            w_hat(t)=U_hat(t);
        else
            w_hat(t)=r_bar*w_hat(t-1)+(U_hat(t)-U_hat(t-1));
        end;
    end;
    r_bar_t=r_bar.^linear_trend;
    results_5=ols(w_hat,r_bar_t);
    resid_square=results_5.resid.^2;
    qLL(i)=r_bar*sum(resid_square)-sum(U_hat.^2);
    disp(i);
end;
%xlswrite('Returns_international_results_1980_2010',qLL(1),...
%    'Granger causality--GFD','ap4');
%xlswrite('Returns_international_results_1980_2010',qLL(2),...
%    'Granger causality--GFD','ap9');
%xlswrite('Returns_international_results_1980_2010',qLL(3),...
%    'Granger causality--GFD','ap14');
%xlswrite('Returns_international_results_1980_2010',qLL(4),...
%    'Granger causality--GFD','ap19');
%xlswrite('Returns_international_results_1980_2010',qLL(5),...
%    'Granger causality--GFD','ap24');
%xlswrite('Returns_international_results_1980_2010',qLL(6),...
%    'Granger causality--GFD','ap29');
%xlswrite('Returns_international_results_1980_2010',qLL(7),...
%    'Granger causality--GFD','ap34');
%xlswrite('Returns_international_results_1980_2010',qLL(8),...
%    'Granger causality--GFD','ap39');
%xlswrite('Returns_international_results_1980_2010',qLL(9),...
%    'Granger causality--GFD','ap44');
%xlswrite('Returns_international_results_1980_2010',qLL(10),...
%    'Granger causality--GFD','ap49');
