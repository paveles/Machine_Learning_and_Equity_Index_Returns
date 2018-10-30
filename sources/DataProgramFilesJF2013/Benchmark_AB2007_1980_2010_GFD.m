%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Benchmark_AB2007_1980_2010_GFD.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 05-15-2012

clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading data, 1980:02-2010:12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Data_1980_2010_equity_premium_GFD;
data_equity_premium=100*data_equity_premium; % expressing in percent
Data_1980_2010_bill;
Data_1980_2010_dividend_yield;
data_dy=log(data_dividend_yield);

%%%%%%%%%%%%%%%%
% OLS estimation
%%%%%%%%%%%%%%%%

Y=data_equity_premium(:,[3 4 10 11]);
X_1=data_bill(:,[3 4 10 11]);
X_2=data_dy(:,[3 4 10 11]);
[T,N]=size(Y);
K=3; % number of RHS variables (including intercept)
results_all=zeros(3,K,N+1);
[results]=Estimate_benchmark_GMM(Y,X_1,X_2);
for i=1:N+1;
    results_all(1:2,:,i)=[results(i,1) results(i,3) 100*results(i,5) ; ...
        results(i,2) results(i,4) results(i,6)];
    disp(i);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimating DGP parameters for wild bootstrap
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mean_Y=mean(Y);
e=zeros(T-1,N);
Theta=zeros(2,N); % intercepts for predictor processes
Phi=zeros(2,2,N); % slope coefficients for predictor processes
v=zeros(T-1,2,N); % fitted residuals for predictor processes
for i=1:N;
    Y_i=Y(2:T,i);
    X_i=[ones(T-1,1) X_1(1:T-1,i) X_2(1:T-1,i)];
    results_i=ols(Y_i,X_i);
    e(:,i)=results_i.resid;
    [Theta_i,Phi_i,v_i]=Perform_AHW_bias_reduction_bivariate_VAR(...
        X_1(:,i),X_2(:,i)); % Amihud et al (2009) reduced-bias estimates
    Theta(:,i)=Theta_i;
    Phi(:,:,i)=Phi_i;
    v(:,:,i)=v_i;
    disp(i);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generating pseudo data using wild boostrap under no predictability
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

B=2000; % number of bootstrap replications
Y_star=zeros(T,N,B);
X_1_star=zeros(T,N,B);
X_2_star=zeros(T,N,B);
for b=1:B;
    Y_star(1,:,b)=Y(1,:);
    X_1_star(1,:,b)=X_1(1,:);
    X_2_star(1,:,b)=X_2(1,:);
    w=randn(T-1,1);
    for t=2:T;
        Y_star(t,:,b)=mean_Y+w(t-1)*e(t-1,:);
        for i=1:N;
            X_1_star(t,i,b)=Theta(1,i)+Phi(1,:,i)*...
                [X_1_star(t-1,i,b) ; X_2_star(t-1,i,b)]+w(t-1)*v(t-1,1,i);
            X_2_star(t,i,b)=Theta(2,i)+Phi(2,:,i)*...
                [X_1_star(t-1,i,b) ; X_2_star(t-1,i,b)]+w(t-1)*v(t-1,2,i);
        end;
    end;
    disp(b);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computing statistics for wild bootstrapped pseudo samples
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

stats_boot=zeros(N+1,3,B);
for b=1:B;
    [results_star]=Estimate_benchmark_GMM(Y_star(:,:,b),X_1_star(:,:,b),...
        X_2_star(:,:,b));
    for i=1:(N+1);
        stats_boot(i,:,b)=[results_star(i,2) results_star(i,4) ...
            results_star(i,6)];
        disp([b i]);
    end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computing wild bootstrapped p-values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:N+1;
    for j=1:K;
        stats_boot_i_j=stats_boot(i,j,:);
        if j==1;
            stats_p_i_j=stats_boot_i_j<results_all(2,j,i);
        else
            stats_p_i_j=stats_boot_i_j>results_all(2,j,i);
        end;
        results_all(3,j,i)=sum(stats_p_i_j)/B;
    end;
    disp(i);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Writing results to spreadsheet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%xlswrite('Returns_international_results_1980_2010',results_all(:,:,5),...
%    'Benchmark--GFD','b52');
