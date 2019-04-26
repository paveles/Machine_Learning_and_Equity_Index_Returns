function [return_exclude]=Compute_return_exclude_last_day(data_daily,no_days)

% Last modified: 05-18-2012

% Computes monthly returns from daily returns with the last no_days of the
% month excluded.
%
% Input
%
% data_daily = matrix of daily data (first column = month, second column = day,
%              third column = year, fourth column = price)
% no_days    = number of days at the end of month to exclude
%
% Output
%
% return_exclude = vector of monthly returns with last no_days excluded

total_return_index_lag=[];
t=1;
while t<=size(data_daily,1)-1;
    if data_daily(t,1)==1 & data_daily(t+1,1)==2;
        total_return_index_lag=[total_return_index_lag ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==2 & data_daily(t+1,1)==3;
        total_return_index_lag=[total_return_index_lag ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==3 & data_daily(t+1,1)==4;
        total_return_index_lag=[total_return_index_lag ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==4 & data_daily(t+1,1)==5;
        total_return_index_lag=[total_return_index_lag ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==5 & data_daily(t+1,1)==6;
        total_return_index_lag=[total_return_index_lag ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==6 & data_daily(t+1,1)==7;
        total_return_index_lag=[total_return_index_lag ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==7 & data_daily(t+1,1)==8;
        total_return_index_lag=[total_return_index_lag ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==8 & data_daily(t+1,1)==9;
        total_return_index_lag=[total_return_index_lag ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==9 & data_daily(t+1,1)==10;
        total_return_index_lag=[total_return_index_lag ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==10 & data_daily(t+1,1)==11;
        total_return_index_lag=[total_return_index_lag ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==11 & data_daily(t+1,1)==12;
        total_return_index_lag=[total_return_index_lag ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==12 & data_daily(t+1,1)==1;
        total_return_index_lag=[total_return_index_lag ; data_daily(t-no_days,end)];
    end;
    t=t+1;
end;
total_return_index=[];
t=30;
while t<size(data_daily,1)-1;
    if data_daily(t,1)==2 & data_daily(t+1,1)==3;
        total_return_index=[total_return_index ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==3 & data_daily(t+1,1)==4;
        total_return_index=[total_return_index ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==4 & data_daily(t+1,1)==5;
        total_return_index=[total_return_index ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==5 & data_daily(t+1,1)==6;
        total_return_index=[total_return_index ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==6 & data_daily(t+1,1)==7;
        total_return_index=[total_return_index ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==7 & data_daily(t+1,1)==8;
        total_return_index=[total_return_index ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==8 & data_daily(t+1,1)==9;
        total_return_index=[total_return_index ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==9 & data_daily(t+1,1)==10;
        total_return_index=[total_return_index ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==10 & data_daily(t+1,1)==11;
        total_return_index=[total_return_index ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==11 & data_daily(t+1,1)==12;
        total_return_index=[total_return_index ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==12 & data_daily(t+1,1)==1;
        total_return_index=[total_return_index ; data_daily(t-no_days,end)];
    elseif data_daily(t,1)==1 & data_daily(t+1,1)==2;
        total_return_index=[total_return_index ; data_daily(t-no_days,end)];
    end;
    t=t+1;
end;
total_return_index=[total_return_index ; data_daily(end,end)];
return_exclude=(total_return_index-total_return_index_lag)./total_return_index_lag;