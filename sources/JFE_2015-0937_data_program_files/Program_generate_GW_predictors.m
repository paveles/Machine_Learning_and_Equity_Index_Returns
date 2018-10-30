%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Program_generate_GW_predictors.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 08-18-2015

clear;

% Load Goyal-Welch predictor data, 1973:01-2014:12

input_file='Returns_short_interest_data.xlsx';
input_sheet='GW variables';

% (1) Log dividend-price ratio

SP=xlsread(input_file,input_sheet,'b1226:b1729');
D12=xlsread(input_file,input_sheet,'c1226:c1729');
log_DP=log(D12./SP);

% (2) Log dividend yield

SP_lag=xlsread(input_file,input_sheet,'b1225:b1728');
log_DY=log(D12./SP_lag);

% (3) Log earnings-price ratio

E12=xlsread(input_file,input_sheet,'d1226:d1729');
log_EP=log(E12./SP);

% (4) Log dividend-payout ratio

log_DE=log(D12./E12);

% (5) Stock excess return volatility (annualized)

SP_R=xlsread(input_file,input_sheet,'q1215:q1729'); % 1972:02-2014:12
R_F_lag=xlsread(input_file,input_sheet,'k1214:k1728'); % 1972:01-2014:11
RVOL=nan(length(SP_R)-11,1); % using Mele (2007) estimator
for t=1:length(RVOL);
    RVOL(t)=mean(abs(SP_R(t:t+11)-R_F_lag(t:t+11)));
end;
RVOL=sqrt(pi/2)*sqrt(12)*RVOL; % 1973:01-2014:12

% (6) Book-to-market ratio

BM=xlsread(input_file,input_sheet,'e1226:e1729');

% (7) Net equity issuance

NTIS=xlsread(input_file,input_sheet,'j1226:j1729');

% (8) Treasury bill rate (annualized)

TBL=xlsread(input_file,input_sheet,'f1226:f1729');

% (9) Long-term yield (annualized)

LTY=xlsread(input_file,input_sheet,'i1226:i1729');

% (10) Long-term return

LTR=xlsread(input_file,input_sheet,'m1226:m1729');

% (11) Term spread (annualized)

TMS=LTY-TBL;

% (12) Default yield spread (annualized)

BAA=xlsread(input_file,input_sheet,'h1226:h1729');
AAA=xlsread(input_file,input_sheet,'g1226:g1729');
DFY=BAA-AAA;

% (13) Default return spread

CORPR=xlsread(input_file,input_sheet,'n1226:n1729');
DFR=CORPR-LTR;

% (14) Inflation (lagged, to account for data release)

INFL_lag=xlsread(input_file,input_sheet,'l1225:l1728');

% Collect economics variables, 1973:01-2014:12

GW=[log_DP log_DY log_EP log_DE RVOL BM NTIS TBL LTY LTR TMS DFY DFR ...
    INFL_lag];
save('Program_generate_GW_predictors.mat','GW');
