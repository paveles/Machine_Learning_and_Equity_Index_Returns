
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin,BaseEstimator


class ToConstantTransformer(BaseEstimator, TransformerMixin):

    # here you define the operation it should perform
    def transform(self, X, y=None, **fit_params):
        return pd.DataFrame(np.ones(len(X)))

    # just return self
    def fit(self, X, y=None, **fit_params):
        return self #pd.DataFrame(np.ones(X.shape[0]), index = X.index)

class ToNumpyTransformer(BaseEstimator, TransformerMixin):

    # here you define the operation it should perform
    def transform(self, X, y=None, **fit_params):
        return pd.DataFrame(X).to_numpy()

    # just return self
    def fit(self, X, y=None, **fit_params):
        return self #pd.DataFrame(np.ones(X.shape[0]), index = X.index)

class DisabledCV:
    def __init__(self):
        self.n_splits = 1

    def split(self, X, y, groups=None):
        yield (np.arange(len(X)), np.arange(len(y)))

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits