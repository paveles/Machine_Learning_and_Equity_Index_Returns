%%%%%%%%%%%%%%%%
% Program_main.m
%%%%%%%%%%%%%%%%

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
r=log(1+R_SP500)-log(1+Rfree_lag);
disp('Equity risk premium, ann % summary stats (mean, vol, Sharpe ratio)');
disp([12*mean(r) sqrt(12)*std(r) sqrt(12)*mean(r)/std(r)]);

% Load Goyal-Welch predictor data, 1973:01-2014:12

load('Program_generate_GW_predictors.mat');
stats_GW=[mean(GW)' median(GW)' prctile(GW,1)' prctile(GW,99)' std(GW)'];
disp('Predictor variables, summary stats');
disp('Mean, median, 1st percentile, 99th percentile, std dev');
disp(stats_GW);
rho_GW=nan(size(GW,2));
for i=1:size(GW,2);
    rho_GW(i)=corr(GW(2:end,i),GW(1:end-1,i));
end;
GW_adjust=GW;
GW_adjust(:,7:9)=-GW(:,7:9);
GW_adjust(:,end)=-GW(:,end);
GW_standardize=zscore(GW_adjust);

% Compute principal components for Goyal-Welch predictors

X=GW_standardize;
X(:,[4 11])=[];
[coeff,score,latent]=pca(X);
PC_GW=zscore(score(:,1:3));

% Load short-interest data, 1973:01-2014:12

input_sheet='Short interest';
EWSI=xlsread(input_file,input_sheet,'b2:b505');
log_EWSI=log(EWSI);

% Perform robust trend test for log(EWSI)

[z_lambda_md]=Compute_z_lambda_md(log_EWSI,1);
z_lambda_md_cv=[1.645 1.96 2.58];
disp('Harvey et al (2007) z-statistics (10%, 5%, 1%)');
disp(z_lambda_md);
disp('Critical values');
disp(z_lambda_md_cv);

% Perform robust breaking trend test for log(EWSI)

[t_lambda]=Compute_t_lambda(log_EWSI);
t_lambda_cv=[2.284, 2.563, 3.135];
disp('Harvey et al (2009) t-statistics (10%, 5%, 1%)');
disp(t_lambda);
disp('Critical values');
disp(t_lambda_cv);

% Perform unit root tests for log(EWSI)

[MZ_GLS,ADF_GLS]=Compute_MZ_ADF_GLS(log_EWSI);
MZ_GLS_cv=[-14.2, -17.3, -23.8];
ADF_GLS_cv=[-2.62, -2.91, -3.42];
disp('Ng & Perron (2001) unit root tests');
disp('MZ-GLS statistic');
disp(MZ_GLS);
disp('Critical values');
disp(MZ_GLS_cv);
disp('ADF-GLS statistic');
disp(ADF_GLS);
disp('Critical values');
disp(ADF_GLS_cv);

% Compute log(EWSI) deviation from linear trend

X_linear=[ones(length(log_EWSI),1) (1:1:length(log_EWSI))'];
results_linear=ols(log_EWSI,X_linear);
SII=zscore(results_linear.resid);
short_interest=[EWSI SII];
stats_short_interest=[mean(short_interest)' median(short_interest)' ...
    prctile(short_interest,1)' prctile(short_interest,99)' ...
    std(short_interest)'];
disp('EWSI & SII, summary stats');
disp('Mean, median, 1st percentile, 99th percentile, std dev');
disp(stats_short_interest);
disp('EWSI decadal means');
decadal_mu=[mean(EWSI(1:12*10)) mean(EWSI(12*10+1:12*20)) ...
    mean(EWSI(12*20+1:12*30)) mean(EWSI(12*30+1:end))];
disp(decadal_mu);
rho_SII=corr(SII(2:end),SII(1:end-1));

% Compute predictor correlation matrix

predictor_correlation=corr([GW SII]);

% Write summary statistics et al to Excel file

output_file='Returns_short_interest_results.xlsx';
output_sheet='Summary statistics';
xlwrite(output_file',stats_GW,output_sheet,'b3');
xlwrite(output_file',rho_GW,output_sheet,'m3');
xlwrite(output_file,stats_short_interest,output_sheet,'b17');
xlwrite(output_file,decadal_mu,output_sheet,'h17');
xlwrite(output_file,rho_SII,output_sheet,'m18');
output_file='Returns_short_interest_results.xlsx';
output_sheet='SII trend analysis';
xlwrite(output_file',[z_lambda_md' z_lambda_md_cv'],output_sheet,'b3');
xlwrite(output_file',[t_lambda' t_lambda_cv'],output_sheet,'e3');
xlwrite(output_file',MZ_GLS,output_sheet,'h3');
xlwrite(output_file',MZ_GLS_cv',output_sheet,'i3');
xlwrite(output_file',ADF_GLS,output_sheet,'k3');
xlwrite(output_file',ADF_GLS_cv',output_sheet,'l3');
output_file='Returns_short_interest_results.xlsx';
output_sheet='Predictor correlations';
xlwrite(output_file,predictor_correlation,output_sheet,'b2');

% Compute cumulative returns

h=[1 3 6 12];
r_h=nan(length(r),length(h));
for j=1:length(h);
    for t=1:length(r)-(h(j)-1);
        r_h(t,j)=mean(r(t:t+(h(j)-1)));
    end;
end;

% Compute in-sample results

beta_hat=nan(size(GW,2)+1,4,length(h));
beta_hat_PC_SII=nan(1,4,length(h));
IVX_Wald=nan(2,length(h));
qLL_hat=nan(length(h),1);
for j=1:length(h);
    for i=1:size(GW,2)+2;

        % Bivariate regression models based on Goyal-Welch predictors

        if i<=size(GW,2);
            X_i_j=[ones(length(r_h)-h(j),1) GW_standardize(1:end-h(j),i)];
            results_i_j=nwest(100*r_h(2:end-(h(j)-1),j),X_i_j,h(j));
            beta_hat(i,:,j)=[results_i_j.beta(2) results_i_j.tstat(2) ...
                nan(1,1) 100*results_i_j.rsqr];

        % Bivariate regression model based on SII

        elseif i==size(GW,2)+1;
            X_i_j=[ones(length(r_h)-h(j),1) -SII(1:end-h(j))];
            results_i_j=nwest(100*r_h(2:end-(h(j)-1),j),X_i_j,h(j));
            beta_hat(i,:,j)=[results_i_j.beta(2) results_i_j.tstat(2) ...
                nan(1,1) 100*results_i_j.rsqr];
            [~,IVX_Wald_i_j,pval_i_j]=Compute_IVX_Wald(r,-SII,h(j),0,0.99);
            IVX_Wald(:,j)=[IVX_Wald_i_j ; pval_i_j];
            y_i_j=r_h(2:end-(h(j)-1),j);
            X_i_j=-SII(1:end-h(j));
            Z_i_j=ones(length(r_h)-h(j),1);
            [qLL_hat_i_j]=Compute_qLL_hat(y_i_j,X_i_j,Z_i_j,h(j));
            qLL_hat(j)=qLL_hat_i_j;
        end;
    end;

    % Multiple regression model based on principal components & SII

    X_PC_j=[ones(length(r_h)-1-(h(j)-1),1) PC_GW(1:end-1-(h(j)-1),:)];
    X_PC_SII_j=[ones(length(r_h)-1-(h(j)-1),1) ...
        PC_GW(1:end-1-(h(j)-1),:) -SII(1:end-1-(h(j)-1))];
    results_PC_j=nwest(100*r_h(2:end-(h(j)-1),j),X_PC_j,h(j));
    results_PC_SII_j=nwest(100*r_h(2:end-(h(j)-1),j),X_PC_SII_j,h(j));
    SSE_reduced_j=sum(results_PC_j.resid.^2);
    SSE_full_j=sum(results_PC_SII_j.resid.^2);
    partial_rsqr_j=(SSE_reduced_j-SSE_full_j)/SSE_reduced_j;
    beta_hat_PC_SII(1,:,j)=[results_PC_SII_j.beta(end) ...
        results_PC_SII_j.tstat(end) nan(1,1) 100*partial_rsqr_j];
end;

% Compute fixed-regressor wild bootstrap p-values

X_sink=[GW_standardize -SII];
X_sink(:,[4 11])=[];
X_sink=[ones(length(r_h),1) X_sink];
results_sink=ols(r_h(2:end,1),X_sink(1:end-1,:));
epsilon_hat=results_sink.resid;
B=1000;
beta_hat_tstat_star=nan(B,size(GW,2)+1,length(h));
beta_hat_PC_SII_tstat_star=nan(B,length(h));
rng('default'); % for reproducability
for b=1:B;
    disp(b);
    u_star_b=randn(length(r_h)-1,1);
    r_star_b=[r(1) ; mean(r)+epsilon_hat.*u_star_b];
    r_h_star_b=nan(length(r),length(h));
    for j=1:length(h);
        for t=1:length(r)-(h(j)-1);
            r_h_star_b(t,j)=mean(r_star_b(t:t+(h(j)-1)));
        end;
    end;
    for j=1:length(h);
        for i=1:size(GW,2)+1;
            if i<=size(GW,2);
                X_i_j=[ones(length(r_h)-1-(h(j)-1),1) ...
                    GW_standardize(1:end-1-(h(j)-1),i)];
                results_i_j_star_b=nwest(...
                    100*r_h_star_b(2:end-(h(j)-1),j),X_i_j,h(j));
                beta_hat_tstat_star(b,i,j)=results_i_j_star_b.tstat(2);
            else
                X_i_j=[ones(length(r_h)-1-(h(j)-1),1) ...
                    -SII(1:end-1-(h(j)-1))];
                results_i_j_star_b=nwest(...
                    100*r_h_star_b(2:end-(h(j)-1),j),X_i_j,h(j));
                beta_hat_tstat_star(b,i,j)=results_i_j_star_b.tstat(2);
            end;
        end;
        X_PC_SII_j=[ones(length(r_h)-1-(h(j)-1),1) ...
            PC_GW(1:end-1-(h(j)-1),:) -SII(1:end-1-(h(j)-1))];
        results_PC_SII_j_star_b=nwest(100*r_h_star_b(2:end-(h(j)-1),j),...
            X_PC_SII_j,h(j));
        beta_hat_PC_SII_tstat_star(b,j)=results_PC_SII_j_star_b.tstat(end);      
    end;
end;
for j=1:length(h);
    for i=1:size(GW,2)+1;
        beta_hat(i,3,j)=sum(beta_hat_tstat_star(:,i,j)>...
            beta_hat(i,2,j))/B;
    end;
    beta_hat_PC_SII(1,3,j)=sum(beta_hat_PC_SII_tstat_star(:,j)>...
        beta_hat_PC_SII(1,2,j))/B;
end;
disp('In-sample results: individual predictors');
disp(round2(beta_hat,0.01));
disp('In-sample results: PC(1:3), SII');
disp(round2(beta_hat_PC_SII,0.01));

% Write in-sample results to Excel file

output_file='Returns_short_interest_results.xlsx';
output_sheet='In-sample PR results';
xlwrite(output_file,beta_hat(:,:,1),output_sheet,'b3');
xlwrite(output_file,beta_hat_PC_SII(1,:,1),output_sheet,'b18');
xlwrite(output_file,IVX_Wald(:,1),output_sheet,'b20');
xlwrite(output_file,qLL_hat(1),output_sheet,'b23');
xlwrite(output_file,beta_hat(:,:,2),output_sheet,'g3');
xlwrite(output_file,beta_hat_PC_SII(1,:,2),output_sheet,'g18');
xlwrite(output_file,IVX_Wald(:,2),output_sheet,'g20');
xlwrite(output_file,qLL_hat(2),output_sheet,'g23');
xlwrite(output_file,beta_hat(:,:,3),output_sheet,'l3');
xlwrite(output_file,beta_hat_PC_SII(1,:,3),output_sheet,'l18');
xlwrite(output_file,IVX_Wald(:,3),output_sheet,'l20');
xlwrite(output_file,qLL_hat(3),output_sheet,'l23');
xlwrite(output_file,beta_hat(:,:,4),output_sheet,'q3');
xlwrite(output_file,beta_hat_PC_SII(1,:,4),output_sheet,'q18');
xlwrite(output_file,IVX_Wald(:,4),output_sheet,'q20');
xlwrite(output_file,qLL_hat(4),output_sheet,'q23');

% Take care of out-of-sample preliminaries

T=length(r);
in_sample_end=1989;
R=(in_sample_end-1972)*12; % in-sample period
P=T-R; % out-of-sample period
FC_PM=nan(P,1);
FC_PR=nan(P,size(GW,2)+1,length(h));
FC_alt_detrend=nan(P,3,length(h));
MA_size=60;

% Compute out-of-sample forecasts

for p=1:P;
    disp(p);

    % Prevailing mean benchmark forecast

    FC_PM(p)=mean(r(1:R+(p-1)));

    % Predictive regression forecasts

    for j=1:length(h);

        % Goyal-Welch predictors

        for i=1:size(GW,2);
            X_i_j_p=[ones(R+(p-1)-h(j),1) GW(1:R+(p-1)-h(j),i)];
            results_i_j_p=ols(r_h(2:R+p-h(j),j),X_i_j_p);
            FC_PR(p,i,j)=[1 GW(R+(p-1),i)]*results_i_j_p.beta;
        end;

        % SII, linear detrending

        X_linear_p=[ones(R+(p-1),1) (1:1:R+(p-1))'];
        results_linear_p=ols(log_EWSI(1:R+(p-1)),X_linear_p);
        SII_p=zscore(results_linear_p.resid);
        X_SII_j_p=[ones(R+(p-1)-h(j),1) SII_p(1:R+(p-1)-h(j))];
        results_SII_j_p=ols(r_h(2:R+p-h(j),j),X_SII_j_p);
        FC_PR(p,size(GW,2)+1,j)=[1 SII_p(end)]*results_SII_j_p.beta;

        % SII, quadratic detrending

        X_quadratic_p=[X_linear_p (1:1:R+(p-1)).^2'];
        results_quadratic_p=ols(log_EWSI(1:R+(p-1)),X_quadratic_p);
        SII_p=zscore(results_quadratic_p.resid);
        X_SII_j_p=[ones(R+(p-1)-h(j),1) SII_p(1:R+(p-1)-h(j))];
        results_SII_j_p=ols(r_h(2:R+p-h(j),j),X_SII_j_p);
        FC_alt_detrend(p,1,j)=[1 SII_p(end)]*results_SII_j_p.beta;

        % SII, cubic detrending

        X_cubic_p=[X_quadratic_p (1:1:R+(p-1)).^3'];    
        results_cubic_p=ols(log_EWSI(1:R+(p-1)),X_cubic_p);
        SII_p=zscore(results_cubic_p.resid);
        X_SII_j_p=[ones(R+(p-1)-h(j),1) SII_p(1:R+(p-1)-h(j))];
        results_SII_j_p=ols(r_h(2:R+p-h(j),j),X_SII_j_p);
        FC_alt_detrend(p,2,j)=[1 SII_p(end)]*results_SII_j_p.beta;

        % SII, stochastic detrending

        SII_p=nan(R+(p-1),1);
        for t=MA_size:R+(p-1);
            SII_p(t)=log_EWSI(t)-mean(log_EWSI(t-MA_size+1:t));
        end;
        SII_p(MA_size:end)=zscore(SII_p(MA_size:end));
        X_SII_j_p=[ones(R+(p-1)-h(j)-(MA_size-1),1) ...
            SII_p(MA_size:R+(p-1)-h(j))];
        results_SII_j_p=ols(r_h(MA_size+1:R+p-h(j),j),X_SII_j_p);
        FC_alt_detrend(p,3,j)=[1 SII_p(end)]*results_SII_j_p.beta;
    end;
end;

% Evaluate forecasts

R2OS_PR=nan(size(GW,2)+1,2,length(h));
R2OS_alt_detrend=nan(size(FC_alt_detrend,2),2,length(h));
lambda=nan(size(GW,2),4,length(h));
for j=1:length(h);
    actual_j=r_h(R+1:end-(h(j)-1),j);
    u_PM_j=actual_j-FC_PM(1:end-(h(j)-1));
    u_PR_j=kron(ones(1,size(FC_PR,2)),actual_j)-FC_PR(1:end-(h(j)-1),:,j);
    u_alt_detrend_j=kron(ones(1,size(FC_alt_detrend,2)),actual_j)-...
        FC_alt_detrend(1:end-(h(j)-1),:,j);
    MSFE_PM_j=mean(u_PM_j.^2);
    MSFE_PR_j=mean(u_PR_j.^2);
    MSFE_alt_detrend_j=mean(u_alt_detrend_j.^2);
    R2OS_PR_j=100*(1-MSFE_PR_j/MSFE_PM_j);
    R2OS_alt_detrend_j=100*(1-MSFE_alt_detrend_j/MSFE_PM_j);
    R2OS_PR(:,1,j)=R2OS_PR_j';
    R2OS_alt_detrend(:,1,j)=R2OS_alt_detrend_j';
    L_j=h(j);
    for i=1:size(GW,2)+1;
        f_CW_i_j=u_PM_j.^2-u_PR_j(:,i).^2+(FC_PM(1:end-(h(j)-1))-...
            FC_PR(1:end-(h(j)-1),i,j)).^2;
        results_CW_i_j=nwest(f_CW_i_j,ones(length(f_CW_i_j),1),h(j));
        R2OS_PR(i,2,j)=results_CW_i_j.tstat;
        if i<=size(GW,2);
            results_HLN_i_j=ols(u_PR_j(:,i),u_PR_j(:,i)-u_PR_j(:,end));
            d_i_j=(u_PR_j(:,i)-u_PR_j(:,end)).*u_PR_j(:,i);
            d_i_j_bar=mean(d_i_j);
            Q_i_j=(1/length(u_PR_j(:,i)))*sum(d_i_j.^2);
            for l=1:L_j;
                Q_i_j=Q_i_j+(1/length(u_PR_j(:,i)))*...
                    (1-(l/(L_j+1)))*sum(d_i_j(l+1:end).*d_i_j(1:end-l));
            end;
            HLN_i_j=sqrt(length(u_PR_j(:,i)))*(Q_i_j^(-0.5))*d_i_j_bar;
            lambda(i,1:2,j)=[results_HLN_i_j.beta HLN_i_j];
            results_HLN_i_j=ols(u_PR_j(:,end),u_PR_j(:,end)-u_PR_j(:,i));
            d_i_j=(u_PR_j(:,end)-u_PR_j(:,i)).*u_PR_j(:,end);
            d_i_j_bar=mean(d_i_j);
            Q_i_j=(1/length(u_PR_j(:,end)))*sum(d_i_j.^2);
            for l=1:L_j;
                Q_i_j=Q_i_j+(1/length(u_PR_j(:,end)))*...
                    (1-(l/(L_j+1)))*sum(d_i_j(l+1:end).*d_i_j(1:end-l));
            end;
            HLN_i_j=sqrt(length(u_PR_j(:,end)))*(Q_i_j^(-0.5))*d_i_j_bar;
            lambda(i,3:4,j)=[results_HLN_i_j.beta HLN_i_j];
        end;
    end;
    for k=1:size(FC_alt_detrend,2);
        f_CW_j_k=u_PM_j.^2-u_alt_detrend_j(:,k).^2+...
            (FC_PM(1:end-(h(j)-1))-FC_alt_detrend(1:end-(h(j)-1),k,j)).^2;
        results_CW_j_k=nwest(f_CW_j_k,ones(length(f_CW_j_k),1),h(j));
        R2OS_alt_detrend(k,2,j)=results_CW_j_k.tstat;
    end;
end;

% Write out-of-sample results to Excel file

output_file='Returns_short_interest_results.xlsx';
output_sheet='R2OS statistics';
xlwrite(output_file,R2OS_PR(:,:,1),output_sheet,'b3');
xlwrite(output_file,R2OS_alt_detrend(:,:,1),output_sheet,'b21');
xlwrite(output_file,R2OS_PR(:,:,2),output_sheet,'e3');
xlwrite(output_file,R2OS_alt_detrend(:,:,2),output_sheet,'e21');
xlwrite(output_file,R2OS_PR(:,:,3),output_sheet,'h3');
xlwrite(output_file,R2OS_alt_detrend(:,:,3),output_sheet,'h21');
xlwrite(output_file,R2OS_PR(:,:,4),output_sheet,'k3');
xlwrite(output_file,R2OS_alt_detrend(:,:,4),output_sheet,'k21');
output_sheet='Forecast encompassing';
xlwrite(output_file,lambda(:,1:2,1),output_sheet,'b4');
xlwrite(output_file,lambda(:,3:4,1),output_sheet,'n4');
xlwrite(output_file,lambda(:,1:2,2),output_sheet,'e4');
xlwrite(output_file,lambda(:,3:4,2),output_sheet,'q4');
xlwrite(output_file,lambda(:,1:2,3),output_sheet,'h4');
xlwrite(output_file,lambda(:,3:4,3),output_sheet,'t4');
xlwrite(output_file,lambda(:,1:2,4),output_sheet,'k4');
xlwrite(output_file,lambda(:,3:4,4),output_sheet,'w4');
