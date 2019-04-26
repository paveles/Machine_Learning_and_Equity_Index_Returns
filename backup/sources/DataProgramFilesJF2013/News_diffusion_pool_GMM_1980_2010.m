%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% News_diffusion_pool_GMM_1980_2010.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
Y_pool=data_equity_premium(2:T,:);
Y_pool=Y_pool(:);
X_bill_pool=data_bill(1:T-1,:);
X_bill_pool=X_bill_pool(:);
X_dy_pool=data_dy(1:T-1,:);
X_dy_pool=X_dy_pool(:);
X_pool=[kron(eye(N),ones(T-1,1)) X_bill_pool X_dy_pool];
results_pool=ols(Y_pool,X_pool);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GMM estimation: first step
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_0=[results_pool.beta ; 0.9 ; 0.8];
K=size(X_USA,2)+(size(X,2)-1)*(N-1)+(size(X,2)+1)*(N-1)+(N-1);
W_1=eye(K);
Prob=conAssign('Compute_objective_news_system_pool',[],[],[],[],[],[],x_0);
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
%    'News-diffusion model','j2:j16');
moments=feval('Compute_moments_news_system_pool',x_1,1,Y_USA,Y,X_USA,X);
W_2=Compute_GMM_weight_matrix(moments,0);
Prob=conAssign('Compute_objective_news_system_pool',[],[],[],[],[],[],x_1);
Prob.user.Y_USA=Y_USA;
Prob.user.Y=Y;
Prob.user.X_USA=X_USA;
Prob.user.X=X;
Prob.user.W=W_2;
Results_step_2=tomRun('ucSolve',Prob);
x_2=Results_step_2.x_k;

%%%%%%%%%%%%%%%%%%%%%%%%
% Computing t-statistics
%%%%%%%%%%%%%%%%%%%%%%%%

%x_2=xlsread('Returns_international_results_1980_2010',...
%    'News-diffusion model','k2:k16');
f_0=feval('Compute_moments_news_system_pool',x_2,2,Y_USA,Y,X_USA,X);
for j=1:length(x_2);
    a=zeros(length(x_2),1);
    eps=max(x_2(j)*1e-4,1e-5);
    a(j)=eps;
    M(:,j)=(feval('Compute_moments_news_system_pool',x_2+a,2,Y_USA,Y,...
        X_USA,X)-f_0)/eps;
end;
V_2=pinv(M'*W_2*M)/(size(Y_USA,1)-1);
SE_2=sqrt(diag(V_2));
results_pool=zeros(3,5);
beta_bill=x_2(N+1);
t_stat_bill=beta_bill/SE_2(N+1);
p_value_bill=norm_cdf(t_stat_bill);
results_pool(:,1)=[beta_bill ; t_stat_bill ; p_value_bill];
beta_dy=x_2(N+2);
t_stat_dy=beta_dy/SE_2(N+2);
p_value_dy=1-norm_cdf(t_stat_dy);
results_pool(:,2)=[beta_dy ; t_stat_dy ; p_value_dy];
lambda=x_2(N+3);
t_stat_lambda=lambda/SE_2(N+3);
p_value_lambda=1-norm_cdf(t_stat_lambda);
results_pool(:,3)=[lambda ; t_stat_lambda ; p_value_lambda];
theta=x_2(N+4);
t_stat_theta=(theta-1)/SE_2(N+4);
p_value_theta=norm_cdf(t_stat_theta);
results_pool(:,4)=[theta ; t_stat_theta ; p_value_theta];
beta_2=(1-theta)*lambda;
g=zeros(size(x_2,1),1);
g(N+3)=(1-x_2(N+4));
g(N+4)=-x_2(N+3);
V_beta=g'*V_2*g;
t_stat_beta_2=beta_2/sqrt(V_beta);
p_value_beta_2=1-norm_cdf(t_stat_beta_2);
results_pool(:,5)=[beta_2 ; t_stat_beta_2 ; p_value_beta_2];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Writing results to spreadsheet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%xlswrite('Returns_international_results_1980_2010',...
%    results_pool,'News-diffusion model','b50');
