%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Program_asset_allocation.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 02-08-2016

clear;

% Include stuff for writing to Excel file (if using Mac)

javaaddpath('poi_library/poi-3.8-20120326.jar');
javaaddpath('poi_library/poi-ooxml-3.8-20120326.jar');
javaaddpath('poi_library/poi-ooxml-schemas-3.8-20120326.jar');
javaaddpath('poi_library/xmlbeans-2.3.0.jar');
javaaddpath('poi_library/dom4j-1.6.1.jar');
javaaddpath('poi_library/stax-api-1.0.1.jar');

% Load equity risk premium data, 1973:01-2014:12

input_file='Returns_short_interest_data.xlsx';
input_sheet='GW variables';
Rfree_lag=xlsread(input_file,input_sheet,'k1225:k1728');
R_SP500=xlsread(input_file,input_sheet,'q1226:q1729');
ER=R_SP500-Rfree_lag;

% Load Goyal-Welch predictor data, 1973:01-2014:12

load('Program_generate_GW_predictors.mat');

% Load short-interest data, 1973:01-2014:12

EWSI=xlsread(input_file,'Short interest','b2:b505');
log_EWSI=log(EWSI);

% Compute cumulative excess returns

h=[1 3 6 12];
ER_h=nan(length(ER),length(h));
R_f_h=nan(length(ER),length(h));
for j=1:length(h);
    for t=1:length(ER)-(h(j)-1);
        ER_h(t,j)=prod(1+R_SP500(t:t+(h(j)-1)))-...
            prod(1+Rfree_lag(t:t+(h(j)-1)));
        R_f_h(t,j)=prod(1+Rfree_lag(t:t+(h(j)-1)))-1;
    end;
end;

% Take care of preliminaries

T=length(ER);
in_sample_end=1989;
R=(in_sample_end-1972)*12; % in-sample period
P=T-R; % out-of-sample period
FC_PM=nan(P,length(h));
w_PM=nan(P,length(h));
R_PM=nan(P,length(h));
ER_PM=nan(P,length(h));
FC_PR=nan(P,size(GW,2)+1,length(h));
w_PR=nan(P,size(GW,2)+1,length(h));
R_PR=nan(P,size(GW,2)+1,length(h));
ER_PR=nan(P,size(GW,2)+1,length(h));
R_BH=nan(P,length(h));
ER_BH=nan(P,length(h));
FC_vol=nan(P,length(h));
window=12*10;
RRA=3;
w_LB=-0.5;
w_UB=1.5;

% Compute out-of-sample forecasts

for p=1:P;
    disp(p);
    for j=1:length(h);

        % Volatility

        if R+p-h(j)<=window-1;
            FC_vol(p,j)=std(ER_h(1:R+p-h(j),j));
        else
            FC_vol(p,j)=std(ER_h(R+p-h(j)-(window-1):R+p-h(j),j));
        end;
        
        % Prevailing mean benchmark

        FC_PM(p,j)=mean(ER_h(1:R+p-h(j),j));

        % Predictive regressions

        for i=1:size(GW,2)+1;
            if i<=size(GW,2);
                X_i_j_p=[ones(R+(p-1)-h(j),1) GW(1:R+(p-1)-h(j),i)];
                results_i_j_p=ols(ER_h(2:R+p-h(j),j),X_i_j_p);
                FC_PR(p,i,j)=[1 GW(R+(p-1),i)]*results_i_j_p.beta;
            else
                trend_p=(1:1:R+(p-1))';
                X_linear_p=[ones(R+(p-1),1) trend_p];
                results_linear_p=ols(log_EWSI(1:R+(p-1)),X_linear_p);
                SII_p=zscore(results_linear_p.resid);
                X_SII_j_p=[ones(R+(p-1)-h(j),1) SII_p(1:R+(p-1)-h(j))];
                results_SII_j_p=ols(ER_h(2:R+p-h(j),j),X_SII_j_p);
                FC_PR(p,i,j)=[1 SII_p(end)]*results_SII_j_p.beta;
            end;
        end;
    end;
end;

% Computing portfolio weights/returns

for j=1:length(h);
    for t=1:P/h(j);
        FC_vol_j_t=FC_vol((t-1)*h(j)+1,j);
        FC_PM_j_t=FC_PM((t-1)*h(j)+1,j);
        w_PM_j_t=(1/RRA)*FC_PM_j_t/FC_vol_j_t^2;
        if w_PM_j_t>w_UB;
            w_PM((t-1)*h(j)+1,j)=w_UB;
        elseif w_PM_j_t<w_LB;
            w_PM((t-1)*h(j)+1,j)=w_LB;
        else
            w_PM((t-1)*h(j)+1,j)=w_PM_j_t;
        end;
        R_PM((t-1)*h(j)+1,j)=R_f_h(R+(t-1)*h(j)+1,j)+...
            w_PM((t-1)*h(j)+1,j)*ER_h(R+(t-1)*h(j)+1,j);
        ER_PM((t-1)*h(j)+1,j)=R_PM((t-1)*h(j)+1,j)-...
            R_f_h(R+(t-1)*h(j)+1,j);
        for i=1:size(FC_PR,2);
            FC_PR_i_j_t=FC_PR((t-1)*h(j)+1,i,j);
            w_PR_i_j_t=(1/RRA)*FC_PR_i_j_t/FC_vol_j_t^2;
            if w_PR_i_j_t>w_UB;
                w_PR((t-1)*h(j)+1,i,j)=w_UB;
            elseif w_PR_i_j_t<w_LB;
                w_PR((t-1)*h(j)+1,i,j)=w_LB;
            else
                w_PR((t-1)*h(j)+1,i,j)=w_PR_i_j_t;
            end;
            R_PR((t-1)*h(j)+1,i,j)=R_f_h(R+(t-1)*h(j)+1,j)+...
                w_PR((t-1)*h(j)+1,i,j)*ER_h(R+(t-1)*h(j)+1,j);
            ER_PR((t-1)*h(j)+1,i,j)=R_PR((t-1)*h(j)+1,i,j)-...
                R_f_h(R+(t-1)*h(j)+1,j);
        end;
        R_BH((t-1)*h(j)+1,j)=R_f_h(R+(t-1)*h(j)+1,j)+...
            ER_h(R+(t-1)*h(j)+1,j);
        ER_BH((t-1)*h(j)+1,j)=ER_h(R+(t-1)*h(j)+1,j);
    end;
end;

% Compute CER gains and Sharpe ratios for full OOS period

CER_gain=nan(size(FC_PR,2)+1,length(h));
Sharpe=nan(size(FC_PR,2)+2,length(h));
for j=1:length(h);
    R_PM_j=R_PM(:,j);
    R_PM_j=R_PM_j(find(isfinite(R_PM_j)));
    ER_PM_j=ER_PM(:,j);
    ER_PM_j=ER_PM_j(find(isfinite(ER_PM_j)));
    CER_PM_j=(12/h(j))*(mean(R_PM_j)-0.5*RRA*std(R_PM_j)^2);
    Sharpe(1,j)=sqrt((12/h(j)))*mean(ER_PM_j)/std(ER_PM_j);
    for i=1:size(FC_PR,2);
        R_PR_i_j=R_PR(:,i,j);
        R_PR_i_j=R_PR_i_j(find(isfinite(R_PR_i_j)));
        ER_PR_i_j=ER_PR(:,i,j);
        ER_PR_i_j=ER_PR_i_j(find(isfinite(ER_PR_i_j)));
        CER_PR_i_j=(12/h(j))*(mean(R_PR_i_j)-0.5*RRA*std(R_PR_i_j)^2);
        CER_gain(i,j)=100*(CER_PR_i_j-CER_PM_j);
        Sharpe(i+1,j)=sqrt((12/h(j)))*mean(ER_PR_i_j)/std(ER_PR_i_j);
    end;
    R_BH_j=R_BH(:,j);
    R_BH_j=R_BH_j(find(isfinite(R_BH_j)));
    ER_BH_j=ER_BH(:,j);
    ER_BH_j=ER_BH_j(find(isfinite(ER_BH_j)));
    CER_BH_j=(12/h(j))*(mean(R_BH_j)-0.5*RRA*std(R_BH_j)^2);
    CER_gain(end,j)=100*(CER_BH_j-CER_PM_j);
    Sharpe(end,j)=sqrt((12/h(j)))*mean(ER_BH_j)/std(ER_BH_j);
end;

% Compute CER gains and Sharpe ratios for Global Financial Crisis period

GFC_start=(2006-1989)*12+1;
CER_gain_GFC=nan(size(FC_PR,2)+1,length(h));
Sharpe_GFC=nan(size(FC_PR,2)+2,length(h));
for j=1:length(h);
    R_PM_j=R_PM(GFC_start:end,j);
    R_PM_j=R_PM_j(find(isfinite(R_PM_j)));
    ER_PM_j=ER_PM(GFC_start:end,j);
    ER_PM_j=ER_PM_j(find(isfinite(ER_PM_j)));
    CER_PM_j=(12/h(j))*(mean(R_PM_j)-0.5*RRA*std(R_PM_j)^2);
    Sharpe_GFC(1,j)=sqrt((12/h(j)))*mean(ER_PM_j)/std(ER_PM_j);
    for i=1:size(FC_PR,2);
        R_PR_i_j=R_PR(GFC_start:end,i,j);
        R_PR_i_j=R_PR_i_j(find(isfinite(R_PR_i_j)));
        ER_PR_i_j=ER_PR(GFC_start:end,i,j);
        ER_PR_i_j=ER_PR_i_j(find(isfinite(ER_PR_i_j)));
        CER_PR_i_j=(12/h(j))*(mean(R_PR_i_j)-0.5*RRA*std(R_PR_i_j)^2);
        CER_gain_GFC(i,j)=100*(CER_PR_i_j-CER_PM_j);
        Sharpe_GFC(i+1,j)=sqrt((12/h(j)))*mean(ER_PR_i_j)/std(ER_PR_i_j);
    end;
    R_BH_j=R_BH(GFC_start:end,j);
    R_BH_j=R_BH_j(find(isfinite(R_BH_j)));
    ER_BH_j=ER_BH(GFC_start:end,j);
    ER_BH_j=ER_BH_j(find(isfinite(ER_BH_j)));
    CER_BH_j=(12/h(j))*(mean(R_BH_j)-0.5*RRA*std(R_BH_j)^2);
    CER_gain_GFC(end,j)=100*(CER_BH_j-CER_PM_j);
    Sharpe_GFC(end,j)=sqrt((12/h(j)))*mean(ER_BH_j)/std(ER_BH_j);
end;

% Compute CER gains and Sharpe ratios for pre-crisis period

CER_gain_pre=nan(size(FC_PR,2)+1,length(h));
Sharpe_pre=nan(size(FC_PR,2)+2,length(h));
for j=1:length(h);
    R_PM_j=R_PM(1:GFC_start-1,j);
    R_PM_j=R_PM_j(find(isfinite(R_PM_j)));
    ER_PM_j=ER_PM(1:GFC_start-1,j);
    ER_PM_j=ER_PM_j(find(isfinite(ER_PM_j)));
    CER_PM_j=(12/h(j))*(mean(R_PM_j)-0.5*RRA*std(R_PM_j)^2);
    Sharpe_pre(1,j)=sqrt((12/h(j)))*mean(ER_PM_j)/std(ER_PM_j);
    for i=1:size(FC_PR,2);
        R_PR_i_j=R_PR(1:GFC_start-1,i,j);
        R_PR_i_j=R_PR_i_j(find(isfinite(R_PR_i_j)));
        ER_PR_i_j=ER_PR(1:GFC_start-1,i,j);
        ER_PR_i_j=ER_PR_i_j(find(isfinite(ER_PR_i_j)));
        CER_PR_i_j=(12/h(j))*(mean(R_PR_i_j)-0.5*RRA*std(R_PR_i_j)^2);
        CER_gain_pre(i,j)=100*(CER_PR_i_j-CER_PM_j);
        Sharpe_pre(i+1,j)=sqrt((12/h(j)))*mean(ER_PR_i_j)/std(ER_PR_i_j);
    end;
    R_BH_j=R_BH(1:GFC_start-1,j);
    R_BH_j=R_BH_j(find(isfinite(R_BH_j)));
    ER_BH_j=ER_BH(1:GFC_start-1,j);
    ER_BH_j=ER_BH_j(find(isfinite(ER_BH_j)));
    CER_BH_j=(12/h(j))*(mean(R_BH_j)-0.5*RRA*std(R_BH_j)^2);
    CER_gain_pre(end,j)=100*(CER_BH_j-CER_PM_j);
    Sharpe_pre(end,j)=sqrt((12/h(j)))*mean(ER_BH_j)/std(ER_BH_j);
end;

% Write results to Excel file

output_file='Returns_short_interest_results.xlsx';
output_sheet='Asset allocation';
xlwrite(output_file',CER_gain,output_sheet,'b6');
xlwrite(output_file',Sharpe,output_sheet,'b25');
xlwrite(output_file',CER_gain_GFC,output_sheet,'g6');
xlwrite(output_file',Sharpe_GFC,output_sheet,'g25');
xlwrite(output_file',CER_gain_pre,output_sheet,'l6');
xlwrite(output_file',Sharpe_pre,output_sheet,'l25');
