function [select_AdENET,beta_AdENET,tuning]=Perform_selection_AdENET(Y,X,nfolds,step_1)

% Last modified: 05-25-2012

% Performs variable selection and shrinkage estimation using the adaptive
% elastic net.Tuning parameters are selected via N-fold cross validation.
%
% Input
%
% Y      = n-vector of response observations
% X      = n-by-m matrix of covariate observations
% nfolds = cross-validation parameter
% step_1 = indicator variables for first-step OLS regression:
%              = 0 for multiple regression (kitchen sink)
%              = 1 for marginal regressions
%
% Output
%
% select_AdENET  = vector of indexed selected covariates
% results_AdENET = N-vector of AdENET estimates
% tuning         = 2-vector of selected alpha, lambda values
%
% References
%
% S. Ghosh (2011), "On the Grouped Selection and Model Complexity of the
% Adaptive Elastic Net," Statistics and Computing, 21(3), 451-462

% H. Zou and T. Hastie (2005), "Regularization and Variable Selection Via
% the Elastic Net," Journal of the Royal Statistical Society B, 67(2)
% 301-320

n=size(Y,1);
if step_1==0;
    results_sink=ols(zscore(Y),zscore(X));
    beta_1=results_sink.beta;
elseif step_1==1;
    beta_1=zeros(size(X,2),1);
    for k=1:size(X,2);
        results_k=ols(zscore(Y),zscore(X(:,k)));
        beta_1(k)=results_k.beta;
    end;
end;
alpha=(0.01:0.01:1)';
gama=(0:0.05:1)';
criterion=zeros(size(gama,1),4);
for t=1:size(gama,1);
    w_hat_t=(abs(beta_1)).^(-gama(t));
    criterion_t=zeros(size(alpha,1),3);
    for j=1:size(alpha,1);
        opts.alpha=alpha(j);
        opts.penalty_factor=w_hat_t;
        options=glmnetSet(opts);
        CVerr_t_j=cvglmnet(X,Y,nfolds,[],'response','gaussian',options,[]);
        criterion_t(j,:)=[min(CVerr_t_j.cvm) alpha(j) ...
            CVerr_t_j.lambda_min];
    end;
    [criterion_t_min,criterion_t_min_index]=min(criterion_t(:,1));
    criterion(t,:)=[criterion_t_min gama(t) ...
        criterion_t(criterion_t_min_index,2:end)];
end;
[xxx,criterion_min_index]=min(criterion(:,1));
tuning=criterion(criterion_min_index,2:end)';
w_hat=(abs(beta_1)).^(-tuning(1));
opts.alpha=tuning(2);
opts.lambda=tuning(3);
opts.penalty_factor=w_hat;
options=glmnetSet(opts);
fit=glmnet(X,Y,'gaussian',options);
select_indicator=fit.beta~=0;
select_AdENET=find(select_indicator);
select_AdENET=sort(select_AdENET);
beta_AdENET=fit.beta;
