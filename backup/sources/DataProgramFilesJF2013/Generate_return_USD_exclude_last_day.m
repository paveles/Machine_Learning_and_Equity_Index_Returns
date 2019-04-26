%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate_return_USD_exclude_last_day.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 05-24-2012

clear;

no_days=1; % number of days at end of month to exclude

%%%%%%%%%%%
% Australia
%%%%%%%%%%%

disp('AUS');
data_daily=xlsread('Returns_international_data_1980_2010',...
    'AUS price index, daily','a2:e7845');
data_daily(:,4)=[];
return_exclude_AUS=Compute_return_exclude_last_day(data_daily,no_days);
xlswrite('Returns_international_data_1980_2010',return_exclude_AUS,...
'Stock return (exclude, USD)','b2');

%%%%%%%%
% Canada
%%%%%%%%

disp('CAN');
data_daily=xlsread('Returns_international_data_1980_2010',...
    'CAN return index, daily','a2:e7786');
data_daily(:,4)=[];
return_exclude_CAN=Compute_return_exclude_last_day(data_daily,no_days);
xlswrite('Returns_international_data_1980_2010',return_exclude_CAN,...
    'Stock return (exclude, USD)','c2');

%%%%%%%%
% France
%%%%%%%%

disp('FRA');
data_daily=xlsread('Returns_international_data_1980_2010',...
    'FRA price index, daily','a2:e7774');
data_daily(:,4)=[];
return_exclude_FRA=Compute_return_exclude_last_day(data_daily,no_days);
xlswrite('Returns_international_data_1980_2010',return_exclude_FRA,...
    'Stock return (exclude, USD)','d2');

%%%%%%%%%
% Germany
%%%%%%%%%

disp('DEU');
data_daily=xlsread('Returns_international_data_1980_2010',...
    'DEU return index, daily','a2:e7946');
data_daily(:,4)=[];
return_exclude_DEU=Compute_return_exclude_last_day(data_daily,no_days);
xlswrite('Returns_international_data_1980_2010',return_exclude_DEU,...
    'Stock return (exclude, USD)','e2');

%%%%%%%
% Italy
%%%%%%%

disp('ITA');
data_daily=xlsread('Returns_international_data_1980_2010',...
    'ITA return index, daily','a2:e7997');
data_daily(:,4)=[];
return_exclude_ITA=Compute_return_exclude_last_day(data_daily,no_days);
xlswrite('Returns_international_data_1980_2010',return_exclude_ITA,...
    'Stock return (exclude, USD)','f2');

%%%%%%%
% Japan
%%%%%%%

disp('JPN');
data_daily=xlsread('Returns_international_data_1980_2010',...
    'JPN price index, daily','a2:e7955');
data_daily(:,4)=[];
return_exclude_JPN=Compute_return_exclude_last_day(data_daily,no_days);
xlswrite('Returns_international_data_1980_2010',return_exclude_JPN,...
    'Stock return (exclude, USD)','g2');

%%%%%%%%%%%%%
% Netherlands
%%%%%%%%%%%%%

disp('NLD');
data_daily=xlsread('Returns_international_data_1980_2010',...
    'NLD return index, daily','a2:e7847');
data_daily(:,4)=[];
return_exclude_NLD=Compute_return_exclude_last_day(data_daily,no_days);
xlswrite('Returns_international_data_1980_2010',return_exclude_NLD,...
    'Stock return (exclude, USD)','h2');

%%%%%%%%
% Sweden
%%%%%%%%

disp('SWE');
data_daily=xlsread('Returns_international_data_1980_2010',...
    'SWE price index, daily','a2:e7752');
data_daily(:,4)=[];
return_exclude_SWE=Compute_return_exclude_last_day(data_daily,no_days);
xlswrite('Returns_international_data_1980_2010',return_exclude_SWE,...
    'Stock return (exclude, USD)','i2');

%%%%%%%%%%%%%
% Switzerland
%%%%%%%%%%%%%

disp('CHE');
data_daily=xlsread('Returns_international_data_1980_2010',...
    'CHE price index, daily','a2:e7761');
data_daily(:,4)=[];
return_exclude_CHE=Compute_return_exclude_last_day(data_daily,no_days);
xlswrite('Returns_international_data_1980_2010',return_exclude_CHE,...
    'Stock return (exclude, USD)','j2');

%%%%%%%%%%%%%%%%
% United Kingdom
%%%%%%%%%%%%%%%%

disp('GBR');
data_daily=xlsread('Returns_international_data_1980_2010',...
    'GBR return index, daily','a2:e7892');
data_daily(:,4)=[];
return_exclude_GBR=Compute_return_exclude_last_day(data_daily,no_days);
xlswrite('Returns_international_data_1980_2010',return_exclude_GBR,...
    'Stock return (exclude, USD)','k2');

%%%%%%%%%%%%%%%
% United States
%%%%%%%%%%%%%%%

disp('USA');
data_daily=xlsread('Returns_international_data_1980_2010',...
    'USA price index, daily','a2:e7825');
data_daily(:,4)=[];
return_exclude_USA=Compute_return_exclude_last_day(data_daily, no_days);
xlswrite('Returns_international_data_1980_2010',return_exclude_USA,...
    'Stock return (exclude, USD)','l2');
