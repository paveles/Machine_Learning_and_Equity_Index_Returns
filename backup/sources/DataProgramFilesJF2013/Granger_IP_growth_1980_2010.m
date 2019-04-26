%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Granger_IP_growth_1980_2010.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 05-19-2012

clear;

%%%%%%%%%%%%%%%
% Preliminaries
%%%%%%%%%%%%%%%

Data_1980_2010_industrial_production_growth;
Y=data_IP_growth;
N=size(Y,2);
max_lag=6;

%%%%%%%%%%%%%%%%%%%%%%%%
% Estimating ARDL models
%%%%%%%%%%%%%%%%%%%%%%%%

results=zeros(N-1,7);
beta=zeros(N-1,max_lag);
for i=1:N-1;
    if i==6; % accounting for Swedish outlies in 1980:05/06
        [lag_orders_i_j]=Select_ARDL_lag_AIC(Y(6:end,i),Y(6:end,end),...
            1,max_lag);
        [results_i_j,beta_i_j]=Estimate_ARDL(Y(6:end,i),Y(6:end,end),...
            lag_orders_i_j);
    else
        [lag_orders_i_j]=Select_ARDL_lag_AIC(Y(:,i),Y(:,end),1,max_lag);
        [results_i_j,beta_i_j]=Estimate_ARDL(Y(:,i),Y(:,end),...
            lag_orders_i_j);
    end;
    results(i,:)=[results_i_j lag_orders_i_j];
    results(i,3)=100*results(i,3);
    beta(i,1:lag_orders_i_j(2))=beta_i_j(lag_orders_i_j(1)+2:end)';
    disp([i results(i,:)]);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Writing results to spreadsheet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xlswrite('Returns_international_results_1980_2010',results(1,:),...
    'Granger causality--IP growth','b3');
xlswrite('Returns_international_results_1980_2010',results(2,:),...
    'Granger causality--IP growth','b5');
xlswrite('Returns_international_results_1980_2010',results(3,:),...
    'Granger causality--IP growth','b7');
xlswrite('Returns_international_results_1980_2010',results(4,:),...
    'Granger causality--IP growth','b9');
xlswrite('Returns_international_results_1980_2010',results(5,:),...
    'Granger causality--IP growth','b11');
xlswrite('Returns_international_results_1980_2010',results(6,:),...
    'Granger causality--IP growth','b13');
xlswrite('Returns_international_results_1980_2010',results(7,:),...
    'Granger causality--IP growth','b15');
