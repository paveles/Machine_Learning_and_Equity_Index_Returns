%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Program_additional_in_sample_PR_results.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
R_VW_SP500=xlsread(input_file,input_sheet,'q1226:q1729');
input_sheet='Returns';
r_VW_SP500=log(1+R_VW_SP500)-log(1+Rfree_lag);
R_EW_SP500=xlsread(input_file,input_sheet,'c2:c505');
r_EW_SP500=log(1+R_EW_SP500)-log(1+Rfree_lag);
R_VW_CRSP=xlsread(input_file,input_sheet,'d2:d505');
r_VW_CRSP=log(1+R_VW_CRSP)-log(1+Rfree_lag);
R_EW_CRSP=xlsread(input_file,input_sheet,'e2:e505');
r_EW_CRSP=log(1+R_EW_CRSP)-log(1+Rfree_lag);
r=[r_VW_SP500 r_EW_SP500 r_VW_CRSP r_EW_CRSP];

% Load Goyal-Welch predictor data, 1973:01-2014:12

load('Program_generate_GW_predictors.mat');
GW_standardize=zscore(GW);

% Load short-interest data, 1973:01-2014:12

input_sheet='Short interest';
EWSI=xlsread(input_file,input_sheet,'b2:b505');
SI_alt=xlsread(input_file,input_sheet,'b2:e505');
log_EWSI=log(EWSI);
log_SI_alt=log(SI_alt);

% Detrend log short interest series

T=length(log_EWSI);
SII_EWSI=nan(T,4);
SII_alt=nan(T,size(log_SI_alt,2));
trend=(1:1:T)';
X_linear=[ones(T,1) trend];
results_linear=ols(log_EWSI,X_linear);
SII_EWSI(:,1)=zscore(results_linear.resid);
for i=1:size(log_SI_alt,2);
    results_linear_i=ols(log_SI_alt(:,i),X_linear);
    SII_alt(:,i)=zscore(results_linear_i.resid);
end;
X_quadratic=[ones(T,1) trend trend.^2];
results_quadratic=ols(log_EWSI,X_quadratic);
SII_EWSI(:,2)=zscore(results_quadratic.resid);
X_cubic=[ones(T,1) trend trend.^2 trend.^3];
results_cubic=ols(log_EWSI,X_cubic);
SII_EWSI(:,3)=zscore(results_cubic.resid);
MA_size=60;
SII_EWSI_SD=nan(T,1);
for t=MA_size:T;
    SII_EWSI_SD(t)=log_EWSI(t)-mean(log_EWSI(t-MA_size+1:t));
end;
SII_EWSI(MA_size:end,4)=zscore(SII_EWSI_SD(MA_size:end));

% Compute cumulative returns

h=[1 3 6 12];
r_h=nan(length(r),length(h),size(r,2));
for j=1:length(h);
    for k=1:size(r,2);
        for t=1:length(r)-(h(j)-1);
            r_h(t,j,k)=mean(r(t:t+(h(j)-1),k));
        end;
    end;
end;

% Define subsamples observations

indicator_sub=zeros(length(r),4);
index_sub_1=(1:1:(1982-1972)*12)'; % 1973:01-1982:12
indicator_sub(index_sub_1,1)=1;
index_sub_2=((1982-1972)*12+1:1:(1992-1972)*12)'; % 1983:01-1992:12
indicator_sub(index_sub_2,2)=1;
index_sub_3=((1992-1972)*12+1:1:(2002-1972)*12)'; % 1991:01-2002:12
indicator_sub(index_sub_3,3)=1;
index_sub_4=((2002-1972)*12+1:1:(2014-1972)*12)'; % 2003:01-2014:12
indicator_sub(index_sub_4,4)=1;

% Estimate predictive regressions

beta_hat_alt_detrend=nan(size(SII_EWSI,2),4,length(h));
beta_hat_alt_return=nan(size(r,2),4,length(h));
beta_hat_alt_SI=nan(size(SII_alt,2),4,length(h));
beta_hat_subsample=nan(size(indicator_sub,2),4,length(h));
for j=1:length(h);

    % Alternative detrending

    for k=1:size(SII_EWSI,2);
        if k==size(SII_EWSI,2);
            X_j_k=[ones(length(r)-(MA_size-1)-h(j),1) ...
                -SII_EWSI(MA_size:end-h(j),k)];
            results_j_k=nwest(100*r_h(MA_size+1:end-(h(j)-1),j,1),...
                X_j_k,h(j));
        else
            X_j_k=[ones(length(r)-h(j),1) -SII_EWSI(1:end-h(j),k)];
            results_j_k=nwest(100*r_h(2:end-(h(j)-1),j,1),X_j_k,h(j));
        end;
        beta_hat_alt_detrend(k,:,j)=[results_j_k.beta(2) ...
            results_j_k.tstat(2) nan(1,1) 100*results_j_k.rsqr];
    end;

    % Alternative market portfolio returns

    for k=1:size(r,2);
        X_j_k=[ones(length(r)-h(j),1) -SII_EWSI(1:end-h(j),1)];
        results_j_k=nwest(100*r_h(2:end-(h(j)-1),j,k),X_j_k,h(j));
        beta_hat_alt_return(k,:,j)=[results_j_k.beta(2) ...
            results_j_k.tstat(2) nan(1,1) 100*results_j_k.rsqr];
    end;

    % Alternative short interest measures

    for k=1:size(SI_alt,2);
        X_j_k=[ones(length(r)-h(j),1) -SII_alt(1:end-h(j),k)];
        results_j_k=nwest(100*r_h(2:end-(h(j)-1),j,1),X_j_k,h(j));
        beta_hat_alt_SI(k,:,j)=[results_j_k.beta(2) ...
            results_j_k.tstat(2) nan(1,1) 100*results_j_k.rsqr];
    end;

    % Subsample analysis

    for i=1:size(indicator_sub,2);
        index_sub_i=find(indicator_sub(:,i)==1);
        X_i_j=[ones(length(index_sub_i),1) -SII_EWSI(index_sub_i,1)];
        y_i_j=r_h(index_sub_i,j,1);
        results_i_j=nwest(100*y_i_j(2:end-(h(j)-1)),...
            X_i_j(1:end-h(j),:),h(j));
        beta_hat_subsample(i,:,j)=[results_i_j.beta(2) ...
            results_i_j.tstat(2) nan(1,1) 100*results_i_j.rsqr];
    end;
end;

% Compute fixed-regressor wild bootstrap p-values

X_sink=[GW_standardize SII_EWSI(:,1)];
X_sink(:,[4 11])=[];
X_sink=[ones(length(r),1) X_sink];
epsilon_hat=nan(length(r_h)-1,size(r,2));
for k=1:size(r,2);
    results_k=ols(r(2:end,k),X_sink(1:end-1,:));
    epsilon_hat(:,k)=results_k.resid;
end;
B=1000;
beta_hat_tstat_alt_detrend_star=nan(B,size(SII_EWSI,2),length(h));
beta_hat_tstat_alt_return_star=nan(B,size(r_h,3),length(h));
beta_hat_tstat_alt_SI_star=nan(B,size(SI_alt,2),length(h));
beta_hat_tstat_subsample_star=nan(B,size(indicator_sub,2),length(h));
rng('default'); % for reproducability
for b=1:B;
    disp(b);
    u_star_b=randn(length(r)-1,1);
    r_star_b=nan(length(r),size(r,2));
    for k=1:size(r,2);
        r_star_b(:,k)=[r(1,k) ; mean(r(:,k))+epsilon_hat(:,k).*u_star_b];
    end;
    r_h_star_b=nan(length(r_h),length(h),size(r,2));
    for j=1:length(h);
        for k=1:size(r,2);
            for t=1:length(r_h)-(h(j)-1);
                r_h_star_b(t,j,k)=mean(r_star_b(t:t+(h(j)-1),k));
            end;
        end;
    end;
    for j=1:length(h);
        for k=1:size(SII_EWSI,2);
            if k==size(SII_EWSI,2);
                X_j_k=[ones(length(r)-(MA_size-1)-h(j),1) ...
                    -SII_EWSI(MA_size:end-h(j),k)];
                results_j_k_star_b=nwest(...
                    100*r_h_star_b(MA_size+1:end-(h(j)-1),j,1),X_j_k,h(j));
            else
                X_j_k=[ones(length(r)-h(j),1) -SII_EWSI(1:end-h(j),k)];
                results_j_k_star_b=nwest(...
                    100*r_h_star_b(2:end-(h(j)-1),j,1),X_j_k,h(j));
            end;
            beta_hat_tstat_alt_detrend_star(b,k,j)=...
                results_j_k_star_b.tstat(2);
        end;
        for k=1:size(r,2);
            X_j_k=[ones(length(r)-h(j),1) -SII_EWSI(1:end-h(j),1)];
            results_j_k_star_b=nwest(...
                100*r_h_star_b(2:end-(h(j)-1),j,k),X_j_k,h(j));
            beta_hat_tstat_alt_return_star(b,k,j)=...
                results_j_k_star_b.tstat(2);
        end;
        for k=1:size(SI_alt,2);
            X_j_k=[ones(length(r)-h(j),1) -SII_alt(1:end-h(j),k)];
            results_j_k_star_b=nwest(...
                100*r_h_star_b(2:end-(h(j)-1),j,1),X_j_k,h(j));
            beta_hat_tstat_alt_SI_star(b,k,j)=...
                results_j_k_star_b.tstat(2);
        end;
        for i=1:size(indicator_sub,2);
            index_sub_i=find(indicator_sub(:,i)==1);
            X_i_j=[ones(length(index_sub_i),1) -SII_EWSI(index_sub_i,1)];
            y_i_j_star_b=r_h_star_b(index_sub_i,j,1);
            results_i_j_star_b=nwest(...
                100*y_i_j_star_b(2:end-(h(j)-1)),X_i_j(1:end-h(j),:),h(j));
            beta_hat_tstat_subsample_star(b,i,j)=...
                results_i_j_star_b.tstat(2);
        end;
    end;
end;
for j=1:length(h);
    for k=1:size(SII_EWSI,2);
        beta_hat_alt_detrend(k,3,j)=...
            sum(beta_hat_tstat_alt_detrend_star(:,k,j)>...
            beta_hat_alt_detrend(k,2,j))/B;
    end;
    for k=1:size(r,2);
        beta_hat_alt_return(k,3,j)=...
            sum(beta_hat_tstat_alt_return_star(:,k,j)>...
            beta_hat_alt_return(k,2,j))/B;
    end;
    for k=1:size(SI_alt,2);
        beta_hat_alt_SI(k,3,j)=...
            sum(beta_hat_tstat_alt_SI_star(:,k,j)>...
            beta_hat_alt_SI(k,2,j))/B;
    end;
    for i=1:size(indicator_sub,2);
        beta_hat_subsample(i,3,j)=...
            sum(beta_hat_tstat_subsample_star(:,i,j)>...
            beta_hat_subsample(i,2,j))/B;
    end;
end;

% Write results to Excel file

output_file='Returns_short_interest_results.xlsx';
output_sheet='Additional in-sample PR results';
xlwrite(output_file',beta_hat_alt_detrend(:,:,1),output_sheet,'e3');
xlwrite(output_file',beta_hat_alt_detrend(:,:,2),output_sheet,'j3');
xlwrite(output_file',beta_hat_alt_detrend(:,:,3),output_sheet,'o3');
xlwrite(output_file',beta_hat_alt_detrend(:,:,4),output_sheet,'t3');
xlwrite(output_file',beta_hat_alt_return(:,:,1),output_sheet,'e8');
xlwrite(output_file',beta_hat_alt_return(:,:,2),output_sheet,'j8');
xlwrite(output_file',beta_hat_alt_return(:,:,3),output_sheet,'o8');
xlwrite(output_file',beta_hat_alt_return(:,:,4),output_sheet,'t8');
xlwrite(output_file',beta_hat_alt_SI(:,:,1),output_sheet,'e13');
xlwrite(output_file',beta_hat_alt_SI(:,:,2),output_sheet,'j13');
xlwrite(output_file',beta_hat_alt_SI(:,:,3),output_sheet,'o13');
xlwrite(output_file',beta_hat_alt_SI(:,:,4),output_sheet,'t13');
xlwrite(output_file',beta_hat_subsample(:,:,1),output_sheet,'e18');
xlwrite(output_file',beta_hat_subsample(:,:,2),output_sheet,'j18');
xlwrite(output_file',beta_hat_subsample(:,:,3),output_sheet,'o18');
xlwrite(output_file',beta_hat_subsample(:,:,4),output_sheet,'t18');
