import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
dat = sm.datasets.get_rdataset("Guerry", "HistData").data
results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()

print(results.summary())
print(results.rsquared)
print(results.rsquared_adj)

#print(results.rsquared_adj())
r2 = results.rsquared
N = 86
K = 2
print(1-(1-r2)*(N-1)/(N-K-1))
