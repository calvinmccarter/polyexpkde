import numpy as np
import scipy.integrate as spig
import scipy.interpolate as spip

from sklearn.base import BaseEstimator
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    check_is_fitted,
    check_random_state,
)

from polyexpkde.ksum import (
    kernel_eval,
    kernel_density,
)

class PolyExpKernelDensity(BaseEstimator):
    r"""Polynomial-exponential univariate kernel density estimation.

    This implements the method described in:
    "Fast exact evaluation of univariate kernel sums" (Hofmeyr, 2019).

    Parameters
    ----------
    bw_method : float > 0, 'silverman' (default='silverman')
        The global bandwidth factor parameter, for computing a baseline
        pilot estimate of the density.
        As with scipy.stats.gaussian_kde, this is multiplied with the
        standard deviation of each feature.

    order : int
        The order of the polynomial for the polynomial-exponential kernel.
    """
    def __init__(self, bw_method="silverman", order=4):
        self.bw_method = bw_method
        self.order = order

        self.X_ = None  # ndarray (n,1)

    def _check_inputs(self, X, in_fit, accept_sparse_negative=False, copy=False):
        """Check inputs before fit and transform."""
        X = self._validate_data(
            X,
            reset=in_fit,
            accept_sparse=False,
            copy=copy,
            dtype=FLOAT_DTYPES,
            force_all_finite=True,
        )

        return X

    def fit(self, X):
        N, D = X.shape
        if D != 1:
            raise NotImplementedError("This KDE is univariate.")
        X = self._check_inputs(X, in_fit=True, copy=False)
        self.X_ = X.copy()

        return self

    def evaluate(self, X):
        check_is_fitted(self)
        N, D = X.shape
        if D != 1:
            raise NotImplementedError("This KDE is univariate.")

        X = self._check_inputs(X, in_fit=False, copy=False)

        X_ = self.X_.flatten()
        X = X.flatten()
        kde = kernel_density(
            x=X_, 
            x_eval=X,
            bw_method=self.bw_method,
            order=self.order,
        )
        kde = kde.reshape(N, D)

        return kde

    def pdf(self, X):
        """Unlike scipy.stats.gaussian_kde, expects 2d"""
        return self.evaluate(X)

    def logpdf(self, X):
        """Unlike scipy.stats.gaussian_kde, expects 2d"""
        return np.log(self.evaluate(X))

    def sample(self, n_samples, random_state=None):
        check_is_fitted(self)
        rng = check_random_state(random_state)

        x_eval = np.linspace(-10., 10., 10000)
        kde = kernel_eval(x_eval, order=self.order)
        cums = spig.cumulative_trapezoid(kde, x_eval, initial=0.)
        inverse_func = spip.interp1d(
            cums, x_eval, bounds_error=False, fill_value="extrapolate")
        unif = rng.uniform(0, 1, size=(n_samples,))
        noise = inverse_func(unif).reshape(-1, 1)
        
        idx = rng.randint(0, self.X_.shape[0], size=(n_samples,))
        S = self.X_[idx] + noise
        return S

    def resample(self, size=None, seed=None):
        if size is None:
            size = self.N_
        return self.sample(size, seed).reshape(1, -1)

    def score_samples(self, X):
        return self.logpdf(X)
 
    def score(self, X):
        log_probs = self.logpdf(X)
        sumlogprobs = np.sum(log_probs)
        return sumlogprobs

    def integrate_box_1d(self, low, high, maxpts=10000):
        check_is_fitted(self)
        xs = np.linspace(start=low, stop=high, num=maxpts)
        ys = self.evaluate(xs.reshape(-1, 1)).flatten()
        cums = spig.cumulative_trapezoid(ys, xs, initial=0.)
        return cums[-1]
