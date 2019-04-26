%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Program_VAR_decomposition.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
R_SP500=xlsread(input_file,input_sheet,'q1226:q1729');
r=log(1+R_SP500);

% Load Goyal-Welch predictor data, 1973:01-2014:12

load('Program_generate_GW_predictors.mat');
log_DP=GW(:,1);
GW_adjust=GW;
GW_adjust(:,7:9)=-GW(:,7:9);
GW_adjust(:,end)=-GW(:,end);
GW_standardize=zscore(GW_adjust);

% Compute principal components for Goyal-Welch predictors

X=GW_standardize;
X(:,[4 11])=[];
[coeff,score,latent]=pca(X);
PC_GW=zscore(score(:,1:3));

% Load short-interest data, 1973:01-2013:12

input_sheet='Short interest';
EWSI=xlsread(input_file,input_sheet,'b2:b505');
log_EWSI=log(EWSI);

% Compute log(EWSI) deviation from linear trend

X_linear=[ones(length(log_EWSI),1) (1:1:length(log_EWSI))'];
results_linear=ols(log_EWSI,X_linear);
SII=zscore(results_linear.resid);

% Estimate predictive regression for log return based on SII

T=length(r);
PR_results=nwest(100*r(2:end),[ones(T-1,1) SII(1:end-1)],1);
beta_hat=[PR_results.beta(2) PR_results.tstat(2)];
disp('Predictive regression results based on SII');
disp(round2(beta_hat,0.01));

% Perform VAR-based return decompositions

rho_hat=1/(1+exp(mean(log_DP)));
E_hat_r=nan(T-1,size(GW,2)+1); % fitted return
eta_hat_DR=nan(T-1,size(GW,2)+1); % discount rte news
eta_hat_CF=nan(T-1,size(GW,2)+1); % cash flow news
beta_hat_E_hat_r=nan(size(GW,2)+1,2);
beta_hat_CF=nan(size(GW,2)+1,2);
beta_hat_DR=nan(size(GW,2)+1,2);
for i=1:size(GW,2)+1;

    % Endogenous variables for VAR

    if i==1;
        Y_i=[r GW(:,1)];
    elseif i>1 && i<size(GW,2)+1;
        Y_i=[r GW(:,1) GW(:,i)];
    elseif i==size(GW,2)+1;
        Y_i=[r GW(:,1) PC_GW];
    end;
    Y_i_dev=Y_i-kron(mean(Y_i),ones(T,1));

    % Estimating VAR model (in deviation form)

    A_hat_i=nan(size(Y_i,2),size(Y_i,2)); % VAR slope parameters
    epsilon_hat_i=nan(length(Y_i_dev)-1,size(Y_i,2)); % VAR shocks
    for j=1:size(Y_i,2);
        results_i_j=ols(Y_i_dev(2:end,j),Y_i_dev(1:end-1,:));
        if j==1;
            E_hat_r_i=results_i_j.yhat;
            E_hat_r(:,i)=E_hat_r_i;
        end;
        A_hat_i(j,:)=results_i_j.beta';
        epsilon_hat_i(:,j)=results_i_j.resid;
    end;

    % Return innovations & news components

    e1_i=[1 ; zeros(size(Y_i,2)-1,1)];
    eta_hat_i=(e1_i'*epsilon_hat_i')';
    eta_hat_DR_i=(e1_i'*rho_hat*A_hat_i*inv(eye(size(Y_i,2))-...
        rho_hat*A_hat_i)*epsilon_hat_i')';
    eta_hat_CF_i=eta_hat_i+eta_hat_DR_i;
    eta_hat_DR(:,i)=100*eta_hat_DR_i;
    eta_hat_CF(:,i)=100*eta_hat_CF_i;

    % Predictive regressions based on SII
    
    results_E_hat_r_i=nwest(100*(E_hat_r_i+mean(r)),...
        [ones(T-1,1) SII(1:end-1)],1);
    beta_hat_E_hat_r(i,:)=[results_E_hat_r_i.beta(2) ...
        results_E_hat_r_i.tstat(2)];
    results_DR_i=nwest(100*eta_hat_DR_i,SII(1:end-1),1);
    beta_hat_DR(i,:)=[results_DR_i.beta results_DR_i.tstat];
    results_CF_i=nwest(100*eta_hat_CF_i,SII(1:end-1),1);
    beta_hat_CF(i,:)=[results_CF_i.beta results_CF_i.tstat];
end;
disp('Component predictive regression results based on SII');
disp(round2(beta_hat_E_hat_r,0.01));
disp(round2(beta_hat_CF,0.01));
disp(round2(beta_hat_DR,0.01));

% Write results to Excel file

output_file='Returns_short_interest_results.xlsx';
output_sheet='VAR decomposition';
xlwrite(output_file',beta_hat,output_sheet,'m19');
xlwrite(output_file',beta_hat_E_hat_r,output_sheet,'b3');
xlwrite(output_file',beta_hat_CF,output_sheet,'e3');
xlwrite(output_file',beta_hat_DR,output_sheet,'h3');
