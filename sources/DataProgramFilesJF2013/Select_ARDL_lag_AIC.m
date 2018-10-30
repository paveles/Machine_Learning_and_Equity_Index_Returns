function [lag_orders_star]=Select_ARDL_lag_AIC(Y,X,h,max_lag)

% Last modified: 05-19-2012

% Selects lag orders for ARDL model using AIC.
%
% Input
%
% Y       = T-vector of dependent variable observations
% X       = T-vector of x variable observations
% h       = forecast horizon
% max_lag = maximum lag order
%
% Output
%
% lag_orders_star = 1-by-2 vector of selected lags

T=size(Y,1);
Y_h=zeros(T-(h-1),1);
for t=1:T-(h-1);
    Y_h(t)=mean(Y(t:t+(h-1)));
end;
Y=Y(1:size(Y_h,1));
X=X(1:size(Y_h,1));
Y_h=Y_h(max_lag+1:end);
lag_orders=[zeros(max_lag,1) (1:1:max_lag)'];
Y_lags=[];
X_lags=[];
for q=1:max_lag;
    lag_orders=[lag_orders ; q*ones(max_lag,1) (1:1:max_lag)'];
    Y_lags=[Y_lags Y(max_lag-(q-1):T-q-(h-1))];
    X_lags=[X_lags X(max_lag-(q-1):T-q-(h-1))];
end;
T=size(Y_h,1);
AIC=[];
for q_1=1:max_lag+1;
    for q_2=1:max_lag;
        if q_1==1;
            RHS=[ones(T,1) X_lags(:,1:q_2)];
        else
            RHS=[ones(T,1) Y_lags(:,1:q_1-1) X_lags(:,1:q_2)];
        end;
        beta=inv(RHS'*RHS)*(RHS'*Y_h);
        e=Y_h-RHS*beta;
        AIC=[AIC ; log(e'*e/T)+2*size(beta,1)/T];
    end;
end;
[xxx,index_min]=min(AIC);
lag_orders_star=lag_orders(index_min,:);
