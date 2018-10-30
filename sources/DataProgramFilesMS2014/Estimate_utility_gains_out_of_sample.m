%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate_utility_gains_out_of_sample.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 09-03-2013

clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading data/defining variables, 1950:12-2010:12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Loading data');
input_file='Returns_econ_tech_results';
input_sheet='Equity premium';
y=xlsread(input_file,input_sheet,'c289:c1021');
input_sheet='Macroeconomic variables';
X_ECON=xlsread(input_file,input_sheet,'b289:o1021');
r_f_lag=xlsread(input_file,input_sheet,'q288:q1020');
input_sheet='Technical indicators';
X_TECH=xlsread(input_file,input_sheet,'b289:o1021');

% Adjusting economic variables where necessary for positive expected slope

X_ECON(:,7)=-X_ECON(:,7);
X_ECON(:,8)=-X_ECON(:,8);
X_ECON(:,9)=-X_ECON(:,9);
X_ECON(:,14)=-X_ECON(:,14);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computing out-of-sample forecasts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Preliminaries

disp('Generating out-of-sample forecasts');
T=size(y,1);
N_ECON=size(X_ECON,2);
N_TECH=size(X_TECH,2);
R=(1965-1950)*12+1; % 1950:12-1965:12 initial in-sample period
P=T-R; % forecast evaluation period starts in 1966:01
actual=nan(P,1);
FC_HA=nan(P,1);
FC_ECON=nan(P,N_ECON+2);
FC_TECH=nan(P,N_TECH+2);
FC_ALL=nan(P,2);
k_max_ECON=3;
k_max_TECH=1;
k_max_ALL=4;
IC=3;
FC_VOL=nan(P,1);
VOL_window=12*5;

% Computing recursive forecasts

for t=1:P;
    actual(t)=y(R+t);
    y_t=y(1:R+(t-1));
    FC_HA(t)=mean(y_t);
    X_ECON_t=X_ECON(1:R+(t-1),:);
    X_TECH_t=X_TECH(1:R+(t-1),:);

    % Bivariate predictive regression forecasts: macroeconomic variables

    for i=1:N_ECON;
        results_ECON_i_t=ols(y_t(2:end),...
            [ones(R+(t-2),1) X_ECON_t(1:end-1,i)]);
        FC_ECON(t,i)=[1 X_ECON_t(end,i)]*results_ECON_i_t.beta;
    end;

    % Bivariate predictive regression forecasts: technical indicators

    for i=1:N_TECH;
        results_TECH_i_t=ols(y_t(2:end),...
            [ones(R+(t-2),1) X_TECH_t(1:end-1,i)]);
        FC_TECH(t,i)=[1 X_TECH_t(end,i)]*results_TECH_i_t.beta;
    end;

    % Additional forecasts based on all macroeconomic variables

    FC_ECON(t,N_ECON+1)=mean(FC_ECON(t,1:N_ECON),2);
    [Lambda_ECON_t,F_hat_ECON_t]=princomp(zscore(X_ECON_t));
    [k_ECON_t]=Perform_selection_IC(y_t(2:end),...
        F_hat_ECON_t(1:end-1,1:k_max_ECON),IC);
    F_hat_ECON_t=zscore(F_hat_ECON_t(:,1:k_ECON_t));
    results_ECON_PC_t=ols(y_t(2:end),...
        [ones(R+(t-2),1) F_hat_ECON_t(1:end-1,:)]);
    FC_ECON(t,N_ECON+2)=[1 F_hat_ECON_t(end,:)]*results_ECON_PC_t.beta;

    % Additional forecasts based on all technical indicators

    FC_TECH(t,N_TECH+1)=mean(FC_TECH(t,1:N_TECH),2);
    [Lambda_TECH_t,F_hat_TECH_t]=princomp(zscore(X_TECH_t));
    [k_TECH_t]=Perform_selection_IC(y_t(2:end),...
        F_hat_TECH_t(1:end-1,1:k_max_TECH),IC);
    F_hat_TECH_t=zscore(F_hat_TECH_t(:,1:k_TECH_t));
    results_TECH_PC_t=ols(y_t(2:end),...
        [ones(R+(t-2),1) F_hat_TECH_t(1:end-1,:)]);
    FC_TECH(t,N_TECH+2)=[1 F_hat_TECH_t(end,:)]*results_TECH_PC_t.beta;

    % Forecasts based on all variables taken together

    FC_ALL(t,1)=mean([FC_ECON(t,1:N_ECON) FC_TECH(t,1:N_TECH)],2);
    [Lambda_ALL_t,F_hat_ALL_t]=princomp(zscore([X_ECON_t X_TECH_t]));
    [k_ALL_t]=Perform_selection_IC(y_t(2:end),...
        F_hat_ALL_t(1:end-1,1:k_max_ALL),IC);
    F_hat_ALL_t=zscore(F_hat_ALL_t(:,1:k_ALL_t));
    results_ALL_PC_t=ols(y_t(2:end),...
        [ones(R+(t-2),1) F_hat_ALL_t(1:end-1,:)]);
    FC_ALL(t,2)=[1 F_hat_ALL_t(end,:)]*results_ALL_PC_t.beta;

    % Volatility forecast

    FC_VOL(t)=mean(y(R+(t-1)-...
        VOL_window+1:R+(t-1)).^2)-...
        (mean(y(R+(t-1)-VOL_window+1:R+(t-1))))^2;
    disp(t);
    disp([actual(t) FC_HA(t) FC_ECON(t,end) FC_TECH(t,end) FC_ALL(t,end)]);
    disp(FC_VOL(t));
end;

%%%%%%%%%%%%%%%%%%%%%%
% Evaluating forecasts
%%%%%%%%%%%%%%%%%%%%%%

% Preliminaries

disp('Evaluating forecasts');
c_bp=0;
results_ECON=nan(size(FC_ECON,2),3);
v_ECON_cycle=nan(size(FC_ECON,2),2);
results_TECH=nan(size(FC_TECH,2),3);
v_TECH_cycle=nan(size(FC_TECH,2),2);
results_ALL=nan(size(FC_ALL,2),3);
v_ALL_cycle=nan(size(FC_ALL,2),2);
y_P=y(R+1:T);
r_f_lag_P=r_f_lag(R+1:T);
gamma_MV=5;
input_sheet='Equity premium';
recession=xlsread(input_file,input_sheet,'d470:d1021');
expansion=ones(size(recession,1),1)-recession;
index_exp=find(expansion);
index_rec=find(recession);

% Computing average utility: historical average

[v_HA,SR_HA,xxx,xxx,TO_HA]=Perform_asset_allocation(...
    y_P,r_f_lag_P,FC_HA,FC_VOL,gamma_MV,c_bp);
results_HA=[1200*v_HA SR_HA 100*TO_HA];
[v_HA_exp,xxx,xxx,xxx,xxx]=Perform_asset_allocation(...
    y_P(index_exp),r_f_lag_P(index_exp),FC_HA(index_exp),...
    FC_VOL(index_exp),gamma_MV,c_bp);
[v_HA_rec,xxx,xxx,xxx,xxx]=Perform_asset_allocation(...
    y_P(index_rec),r_f_lag_P(index_rec),FC_HA(index_rec),...
    FC_VOL(index_rec),gamma_MV,c_bp);

% Computing average utility gains: macroeconomic variables

for i=1:size(FC_ECON,2);
    [v_ECON_i,SR_ECON_i,xxx,xxx,TO_ECON_i]=Perform_asset_allocation(...
        y_P,r_f_lag_P,FC_ECON(:,i),FC_VOL,gamma_MV,c_bp);
    results_ECON(i,:)=[1200*(v_ECON_i-v_HA) SR_ECON_i ...
        TO_ECON_i/TO_HA];
    [v_ECON_i_exp,xxx,xxx,xxx,xxx]=Perform_asset_allocation(...
        y_P(index_exp),r_f_lag_P(index_exp),FC_ECON(index_exp,i),...
        FC_VOL(index_exp),gamma_MV,c_bp);
    [v_ECON_i_rec,xxx,xxx,xxx,xxx]=Perform_asset_allocation(...
        y_P(index_rec),r_f_lag_P(index_rec),FC_ECON(index_rec,i),...
        FC_VOL(index_rec),gamma_MV,c_bp);
    v_ECON_cycle(i,:)=[1200*(v_ECON_i_exp-v_HA_exp) ...
        1200*(v_ECON_i_rec-v_HA_rec)];
end;

% Computing average utility gains: technical indicators

for i=1:size(FC_TECH,2);
    [v_TECH_i,SR_TECH_i,xxx,xxx,TO_TECH_i]=Perform_asset_allocation(...
        y_P,r_f_lag_P,FC_TECH(:,i),FC_VOL,gamma_MV,c_bp);
    results_TECH(i,:)=[1200*(v_TECH_i-v_HA) SR_TECH_i ...
        TO_TECH_i/TO_HA];
    [v_TECH_i_exp,xxx,xxx,xxx,xxx]=Perform_asset_allocation(...
        y_P(index_exp),r_f_lag_P(index_exp),FC_TECH(index_exp,i),...
        FC_VOL(index_exp),gamma_MV,c_bp);
    [v_TECH_i_rec,xxx,xxx,xxx,xxx]=Perform_asset_allocation(...
        y_P(index_rec),r_f_lag_P(index_rec),FC_TECH(index_rec,i),...
        FC_VOL(index_rec),gamma_MV,c_bp);
    v_TECH_cycle(i,:)=[1200*(v_TECH_i_exp-v_HA_exp) ...
        1200*(v_TECH_i_rec-v_HA_rec)];
end;

% Computing average utility gains, all variables taken together

for i=1:size(FC_ALL,2);
    [v_ALL_i,SR_ALL_i,xxx,xxx,TO_ALL_i]=Perform_asset_allocation(...
        y_P,r_f_lag_P,FC_ALL(:,i),FC_VOL,gamma_MV,c_bp);
    results_ALL(i,:)=[1200*(v_ALL_i-v_HA) SR_ALL_i TO_ALL_i/TO_HA];
    [v_ALL_i_exp,xxx,xxx,xxx,xxx]=Perform_asset_allocation(...
        y_P(index_exp),r_f_lag_P(index_exp),FC_ALL(index_exp,i),...
        FC_VOL(index_exp),gamma_MV,c_bp);
    [v_ALL_i_rec,xxx,xxx,xxx,xxx]=Perform_asset_allocation(...
        y_P(index_rec),r_f_lag_P(index_rec),FC_ALL(index_rec,i),...
        FC_VOL(index_rec),gamma_MV,c_bp);
    v_ALL_cycle(i,:)=[1200*(v_ALL_i_exp-v_HA_exp) ...
        1200*(v_ALL_i_rec-v_HA_rec)];
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Writing results to spreadsheet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Writing results to spreahdsheet');
output_file='Returns_econ_tech_results';
output_sheet='Out-of-sample results';

% Historical average

xlswrite(output_file,results_HA,output_sheet,'h4');

% Forecasts based on macroeconomic variables

xlswrite(output_file,results_ECON,output_sheet,'h8');
xlswrite(output_file,v_ECON_cycle(:,1),output_sheet,'p8');
xlswrite(output_file,v_ECON_cycle(:,2),output_sheet,'u8');

% Forecasts based on technical indicators

xlswrite(output_file,results_TECH,output_sheet,'h27');
xlswrite(output_file,v_TECH_cycle(:,1),output_sheet,'p27');
xlswrite(output_file,v_TECH_cycle(:,2),output_sheet,'u27');

% Forecasts based on all variables taken together

xlswrite(output_file,results_ALL,output_sheet,'h44');
xlswrite(output_file,v_ALL_cycle(:,1),output_sheet,'p44');
xlswrite(output_file,v_ALL_cycle(:,2),output_sheet,'u44');
