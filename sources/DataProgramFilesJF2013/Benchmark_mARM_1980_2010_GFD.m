%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Benchmark_mARM_1980_2010_GFD.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 05-16-2012

clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading data, 1980:02-2010:12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Data_1980_2010_equity_premium_GFD;
data_equity_premium=100*data_equity_premium; % expressing in percent
Data_1980_2010_bill;
Data_1980_2010_dividend_yield;
data_dy=log(data_dividend_yield);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Amihud et al (2009) mARM estimation of benchmark predictive regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Y=data_equity_premium;
X_1=data_bill;
X_2=data_dy;
[T,N]=size(Y);
K=3; % number of RHS variables (including intercept)
beta_c_all=zeros(K-1,N);
tstat_beta_c_all=zeros(K-1,N);
tstat_beta_p_c_all=zeros(K-1,N);
phi_c_all=zeros(K-1,N);
tstat_phi_c_all=zeros(K-1,N);
VAR_c_all=zeros(K-1,K,N);
for i=1:N;
    [beta_i,SE_beta_i,tstat_beta_i,phi_i,tstat_phi_i,VAR_i]=...
        Perform_mARM_2_predictors(Y(:,i),X_1(:,i),X_2(:,i));
    beta_c_all(:,i)=beta_i;
    tstat_beta_c_all(:,i)=tstat_beta_i;
    tstat_beta_p_c_all(1,i)=norm_cdf(tstat_beta_i(1),0,1);
    tstat_beta_p_c_all(2,i)=1-norm_cdf(tstat_beta_i(2),0,1);
    phi_c_all(:,i)=phi_i;
    tstat_phi_c_all(:,i)=tstat_phi_i;
    VAR_c_all(:,:,i)=VAR_i;
    disp(i);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Writing results to spreadsheet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Australia

%xlswrite('Returns_international_results_1980_2010',[beta_c_all(:,1)' ...
%    phi_c_all(:,1)' ; tstat_beta_c_all(:,1)' tstat_phi_c_all(:,1)'],...
%    'Benchmark--GFD','j4');
%xlswrite('Returns_international_results_1980_2010',...
%    tstat_beta_p_c_all(:,1)','Benchmark--GFD','j6');
%xlswrite('Returns_international_results_1980_2010',...
%    [VAR_c_all(1,:,1) VAR_c_all(2,:,1)],'Benchmark--GFD','o4');

% Canada

%xlswrite('Returns_international_results_1980_2010',[beta_c_all(:,2)' ...
%    phi_c_all(:,2)' ; tstat_beta_c_all(:,2)' tstat_phi_c_all(:,2)'],...
%    'Benchmark--GFD','j8');
%xlswrite('Returns_international_results_1980_2010',...
%    tstat_beta_p_c_all(:,2)','Benchmark--GFD','j10');
%xlswrite('Returns_international_results_1980_2010',...
%    [VAR_c_all(1,:,2) VAR_c_all(2,:,2)],'Benchmark--GFD','o8');

% France

%xlswrite('Returns_international_results_1980_2010',[beta_c_all(:,3)' ...
%    phi_c_all(:,3)' ; tstat_beta_c_all(:,3)' tstat_phi_c_all(:,3)'],...
%    'Benchmark--GFD','j12');
%xlswrite('Returns_international_results_1980_2010',...
%    tstat_beta_p_c_all(:,3)','Benchmark--GFD','j14');
%xlswrite('Returns_international_results_1980_2010',...
%    [VAR_c_all(1,:,3) VAR_c_all(2,:,3)],'Benchmark--GFD','o12');

% Germany

%xlswrite('Returns_international_results_1980_2010',[beta_c_all(:,4)' ...
%    phi_c_all(:,4)' ; tstat_beta_c_all(:,4)' tstat_phi_c_all(:,4)'],...
%    'Benchmark--GFD','j16');
%xlswrite('Returns_international_results_1980_2010',...
%    tstat_beta_p_c_all(:,4)','Benchmark--GFD','j18');
%xlswrite('Returns_international_results_1980_2010',...
%    [VAR_c_all(1,:,4) VAR_c_all(2,:,4)],'Benchmark--GFD','o16');

% Italy

%xlswrite('Returns_international_results_1980_2010',[beta_c_all(:,5)' ...
%    phi_c_all(:,5)' ; tstat_beta_c_all(:,5)' tstat_phi_c_all(:,5)'],...
%    'Benchmark--GFD','j20');
%xlswrite('Returns_international_results_1980_2010',...
%    tstat_beta_p_c_all(:,5)','Benchmark--GFD','j22');
%xlswrite('Returns_international_results_1980_2010',...
%    [VAR_c_all(1,:,5) VAR_c_all(2,:,5)],'Benchmark--GFD','o20');

% Japan

%xlswrite('Returns_international_results_1980_2010',[beta_c_all(:,6)' ...
%    phi_c_all(:,6)' ; tstat_beta_c_all(:,6)' tstat_phi_c_all(:,6)'],...
%    'Benchmark--GFD','j24');
%xlswrite('Returns_international_results_1980_2010',...
%    tstat_beta_p_c_all(:,6)','Benchmark--GFD','j26');
%xlswrite('Returns_international_results_1980_2010',...
%    [VAR_c_all(1,:,6) VAR_c_all(2,:,6)],'Benchmark--GFD','o24');

% Netherlands

%xlswrite('Returns_international_results_1980_2010',[beta_c_all(:,7)' ...
%    phi_c_all(:,7)' ; tstat_beta_c_all(:,7)' tstat_phi_c_all(:,7)'],...
%    'Benchmark--GFD','j28');
%xlswrite('Returns_international_results_1980_2010',...
%    tstat_beta_p_c_all(:,7)','Benchmark--GFD','j30');
%xlswrite('Returns_international_results_1980_2010',...
%    [VAR_c_all(1,:,7) VAR_c_all(2,:,7)],'Benchmark--GFD','o28');

% Sweden

%xlswrite('Returns_international_results_1980_2010',[beta_c_all(:,8)' ...
%    phi_c_all(:,8)' ; tstat_beta_c_all(:,8)' tstat_phi_c_all(:,8)'],...
%    'Benchmark--GFD','j32');
%xlswrite('Returns_international_results_1980_2010',...
%    tstat_beta_p_c_all(:,8)','Benchmark--GFD','j34');
%xlswrite('Returns_international_results_1980_2010',...
%    [VAR_c_all(1,:,8) VAR_c_all(2,:,8)],'Benchmark--GFD','o32');

% Switzerland

%xlswrite('Returns_international_results_1980_2010',[beta_c_all(:,9)' ...
%    phi_c_all(:,9)' ; tstat_beta_c_all(:,9)' tstat_phi_c_all(:,9)'],...
%    'Benchmark--GFD','j36');
%xlswrite('Returns_international_results_1980_2010',...
%    tstat_beta_p_c_all(:,9)','Benchmark--GFD','j38');
%xlswrite('Returns_international_results_1980_2010',...
%    [VAR_c_all(1,:,9) VAR_c_all(2,:,9)],'Benchmark--GFD','o36');

% United Kingdom

%xlswrite('Returns_international_results_1980_2010',[beta_c_all(:,10)' ...
%    phi_c_all(:,10)' ; tstat_beta_c_all(:,10)' tstat_phi_c_all(:,10)'],...
%    'Benchmark--GFD','j40');
%xlswrite('Returns_international_results_1980_2010',...
%    tstat_beta_p_c_all(:,10)','Benchmark--GFD','j42');
%xlswrite('Returns_international_results_1980_2010',...
%    [VAR_c_all(1,:,10) VAR_c_all(2,:,10)],'Benchmark--GFD','o40');

% United States

%xlswrite('Returns_international_results_1980_2010',[beta_c_all(:,11)' ...
%    phi_c_all(:,11)' ; tstat_beta_c_all(:,11)' tstat_phi_c_all(:,11)'],...
%    'Benchmark--GFD','j44');
%xlswrite('Returns_international_results_1980_2010',...
%    tstat_beta_p_c_all(:,11)','Benchmark--GFD','j46');
%xlswrite('Returns_international_results_1980_2010',...
%    [VAR_c_all(1,:,11) VAR_c_all(2,:,11)],'Benchmark--GFD','o44');
