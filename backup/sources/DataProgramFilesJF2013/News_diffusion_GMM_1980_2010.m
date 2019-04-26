%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% News_diffusion_GMM_1980_2010.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Last modified: 05-25-2012

clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading data, 1980:02-2010:12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Data_1980_2010_equity_premium_GFD;
data_equity_premium=100*data_equity_premium; % expressing in percent
Data_1980_2010_bill;
Data_1980_2010_dividend_yield;
data_dy=log(data_dividend_yield);
Data_1980_2010_equity_premium_GFD_exclude_last_day;
data_equity_premium_exclude=100*data_equity_premium_exclude;
[T,N]=size(data_equity_premium);

%%%%%%%%%%%%%%%%%%%
% Defining matrices
%%%%%%%%%%%%%%%%%%%

Y_USA=data_equity_premium_exclude(:,N);
X_USA=[ones(size(Y_USA,1),1) data_bill(:,N) data_dy(:,N)];
Y=zeros(T,N-1);
X=zeros(T,3,N-1);
for i=1:N-1;
    Y(:,i)=data_equity_premium(:,i);
    X(:,:,i)=[ones(T,1) data_bill(:,i) data_dy(:,i)];
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GMM estimation: first step
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_0=zeros(3*N+2*(N-1),1);
results_USA=ols(Y_USA(2:end),X_USA(1:end-1,:));
x_0(1:3)=results_USA.beta;
for i=1:N-1;
    results_i=ols(Y(2:end,i),X(1:end-1,:,i));
    x_0(3+5*(i-1)+1:3+5*i)=[results_i.beta ; 0.9 ; 0.8];
end;
K=size(X_USA,2)+(size(X,2)-1)*(N-1)+(size(X,2)+1)*(N-1)+(N-1);
W_1=eye(K);
Prob=conAssign('Compute_objective_news_system',[],[],[],[],[],[],x_0);
Prob.user.Y_USA=Y_USA;
Prob.user.Y=Y;
Prob.user.X_USA=X_USA;
Prob.user.X=X;
Prob.user.W=W_1;
Results_step_1=tomRun('ucSolve',Prob);
x_1=Results_step_1.x_k;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GMM estimation: second step
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%x_1=xlsread('Returns_international_results_1980_2010',...
%    'News-diffusion model','h2:h54');
moments=feval('Compute_moments_news_system',x_1,1,Y_USA,Y,X_USA,X);
W_2=Compute_GMM_weight_matrix(moments,0);
Prob=conAssign('Compute_objective_news_system',[],[],[],[],[],[],x_1);
Prob.user.Y_USA=Y_USA;
Prob.user.Y=Y;
Prob.user.X_USA=X_USA;
Prob.user.X=X;
Prob.user.W=W_2;
Results_step_2=tomRun('ucSolve',Prob);
x_2=Results_step_2.x_k;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computing t-statistics/J-statistic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%x_2=xlsread('Returns_international_results_1980_2010',...
%    'News-diffusion model','i2:i54');
f_0=feval('Compute_moments_news_system',x_2,2,Y_USA,Y,X_USA,X);
for j=1:length(x_2);
    a=zeros(length(x_2),1);
    eps=max(x_2(j)*1e-4,1e-5);
    a(j)=eps;
    M(:,j)=(feval('Compute_moments_news_system',x_2+a,2,Y_USA,Y,...
        X_USA,X)-f_0)/eps;
end;
V_2=pinv(M'*W_2*M)/(size(Y_USA,1)-1);
SE_2=sqrt(diag(V_2));
results_slope=zeros(3,size(X_USA,2)+1,N);
for i=1:N-1;
    slope_i=x_2(size(X_USA,2)+(size(X,2)+2)*(i-1)+2:size(X_USA,2)+...
        (size(X,2)+2)*(i-1)+size(X,2)+2);
    results_slope(1,:,i)=slope_i';
    SE_slope_i=SE_2(size(X_USA,2)+(size(X,2)+2)*(i-1)+2:size(X_USA,2)+...
        (size(X,2)+2)*(i-1)+size(X,2)+2);
    tstat_slope_i=[slope_i(1:3)./SE_slope_i(1:3) ; ...
        (slope_i(4)-1)/SE_slope_i(4)];
    results_slope(2,:,i)=tstat_slope_i';
    p_tstat_slope_i=[norm_cdf(tstat_slope_i(1)) ...
        1-norm_cdf(tstat_slope_i(2)) 1-norm_cdf(tstat_slope_i(3)) ...
        norm_cdf(tstat_slope_i(4))];
    results_slope(3,:,i)=p_tstat_slope_i;
end;
slope_USA=x_2(2:size(X_USA,2));
SE_slope_USA=SE_2(2:size(X_USA,2));
tstat_slope_USA=slope_USA./SE_slope_USA;
p_tstat_slope_USA=[norm_cdf(tstat_slope_USA(1)) ...
    1-norm_cdf(tstat_slope_USA(2))];
results_slope(:,1:size(X_USA,2)-1,N)=[slope_USA' ; tstat_slope_USA' ; ...
    p_tstat_slope_USA];
f_2=feval('Compute_objective_news_system',x_2,Prob);
chi2_2=(size(Y_USA,1)-1)*f_2;
p_chi2_2=1-chi2cdf(chi2_2,size(W_2,1)-length(x_2));
J_2=[chi2_2 ; p_chi2_2];
beta_2=zeros(3,N-1);
for i=1:N-1;
    beta_2(1,i)=(1-x_2(3+(i-1)*5+5))*x_2(3+(i-1)*5+4);
    g=zeros(size(x_2,1),1);
    g(3+(i-1)*5+4)=(1-x_2(3+(i-1)*5+5));
    g(3+(i-1)*5+5)=-x_2(3+(i-1)*5+4);
    V_beta=g'*V_2*g;
    beta_2(2,i)=beta_2(1,i)/sqrt(V_beta);
    beta_2(3,i)=1-norm_cdf(beta_2(2,i));
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Writing results to spreadsheet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%xlswrite('Returns_international_results_1980_2010',...
%    [results_slope(:,:,1) beta_2(:,1)],'News-diffusion model','b3');
%xlswrite('Returns_international_results_1980_2010',...
%    [results_slope(:,:,2) beta_2(:,2)],'News-diffusion model','b7');
%xlswrite('Returns_international_results_1980_2010',...
%    [results_slope(:,:,3) beta_2(:,3)],'News-diffusion model','b11');
%xlswrite('Returns_international_results_1980_2010',...
%    [results_slope(:,:,4) beta_2(:,4)],'News-diffusion model','b15');
%xlswrite('Returns_international_results_1980_2010',...
%    [results_slope(:,:,5) beta_2(:,5)],'News-diffusion model','b19');
%xlswrite('Returns_international_results_1980_2010',...
%    [results_slope(:,:,6) beta_2(:,6)],'News-diffusion model','b23');
%xlswrite('Returns_international_results_1980_2010',...
%    [results_slope(:,:,7) beta_2(:,7)],'News-diffusion model','b27');
%xlswrite('Returns_international_results_1980_2010',...
%    [results_slope(:,:,8) beta_2(:,8)],'News-diffusion model','b31');
%xlswrite('Returns_international_results_1980_2010',...
%    [results_slope(:,:,9) beta_2(:,9)],'News-diffusion model','b35');
%xlswrite('Returns_international_results_1980_2010',...
%    [results_slope(:,:,10) beta_2(:,10)],'News-diffusion model','b39');
%xlswrite('Returns_international_results_1980_2010',...
%    results_slope(:,1:2,11),'News-diffusion model','b43');
%xlswrite('Returns_international_results_1980_2010',J_2,...
%    'News-diffusion model','b47');
