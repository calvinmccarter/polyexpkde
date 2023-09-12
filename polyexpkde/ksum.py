from numba import njit
import numpy as np

from scipy.special import factorial

K_EPS = 1e-300


def norm_const_K(betas):
    factorial_terms = np.array([
        betas[k - 1] * factorial(k-1) for k in range(1, len(betas) + 1)
    ])
    normalization_constant = 2 * np.sum(factorial_terms)
    return normalization_constant


def betas_for_order(order):
    unnorm_betas = 1 / factorial(np.arange(order + 1))
    betas = unnorm_betas / norm_const_K(unnorm_betas)
    return betas


def roughness_K(betas):
    beta_use = betas / norm_const_K(betas)
    betakj = np.outer(beta_use, beta_use)
    n = len(betas)
    kpj = np.log2(np.outer(2 ** np.arange(n), 2 ** np.arange(n))).astype(np.int64)
    result = np.sum(betakj / (2 ** kpj) * factorial(kpj))
    return result


def var_K(betas):
    beta_use = betas / norm_const_K(betas)
    factorial_terms = np.array([
        beta_use[k - 1] * factorial(k+1) for k in range(1, len(betas) + 1)
    ])
    return 2 * np.sum(factorial_terms)


def norm_K(betas):
    return np.sqrt(roughness_K(betas))


def factor_Gauss_to_K(betas):
    ret = (roughness_K(betas)/(var_K(betas)**2)*2*np.sqrt(np.pi)) ** 0.2
    return ret


def h_Gauss_to_K(h, betas):
    """Converts bandwidth of Gaussian kernel to that of poly-exp kernel."""
    return h * factor_Gauss_to_K(betas)


def h_K_to_Gauss(h, betas):
    """Converts bandwidth of poly-exp kernel to that of Gaussian kernel."""
    return h / factor_Gauss_to_K(betas)


def silverman(x, betas):
    n, = x.shape
    h = (
        8 * np.sqrt(np.pi) / 3 * roughness_K(betas) / (var_K(betas) ** 2) / n
    ) ** 0.2 * np.std(x)
    return h


@njit(error_model="numpy",cache=True)
def ksum_numba(x, y, x_eval, h, betas, output, counts, coefs, Ly, Ry):
    """Implements kernel density estimation with poly-exponential kernel.

    See https://github.com/DavidHofmeyr/FKSUM and
    "Fast exact evaluation of univariate kernel sums" (Hofmeyr, 2019).

    Parameters
    ----------
    x : ndarray of shape (n,)
        Training data (kernel locations). Usually the observed sample, 
        or bin locations for binned summing. Should be sorted in
        non-decreasing order.

    y : ndarray of shape (n,)
        Corresponding coefficients to be added in kernel weighted sums.
        These will typically be ones if one is not doing binned summing.

    x_eval : ndarray of shape (n_eval,)
        Points at which to evaluate the sums. Should be sorted in
        non-decreasing order.

    h : float > 0
        Bandwidth value.

    betas : ndarray of shape (order + 1,)
        Vector of polynomial-exponential kernel coefficients.

    counts : ndarray of shape (n_eval,)
        Allocated workspace for numba, input values ignored.
        This function computes the location of x_eval w.r.t. x. 
        That is, counts[i] is the number of values in x
        less than or equal to x_eval[i].

    coefs : ndarray of shape (order + 1,)
        Allocated workspace for numba, input values ignored.
        
    Ly : ndarray of shape(order + 1, n)
        Allocated workspace for numba, input values ignored.

    Ry : ndarray of shape(order + 1, n)
        Allocated workspace for numba, input values ignored.
    """
    n = x.shape[0]
    n_eval = x_eval.shape[0]
    order = betas.shape[0] - 1
    output[:] = 0.
    counts[:] = 0
    coefs[:] = 0.
    Ly[:, :] = 0.
    Ry[:, :] = 0.

    for i in range(order + 1):
        Ly[i, 0] = np.power(-x[0], i) * y[0]
    for i in range(1, n):
        for j in range(order + 1):
            Ly[j, i] = (
                np.power(-x[i], j) * y[i] 
                + np.exp((x[i-1] - x[i]) / h) * Ly[j, i-1]
            )
            Ry[j, n - i - 1] = (
                np.exp((x[n - i - 1] - x[n - i]) / h) 
                * (np.power(x[n - i], j) * y[n-i] + Ry[j, n - i])
            )
 
    count = 0
    for i in range(n_eval):
        if x_eval[i] >= x[n - 1]:
            counts[i] = n
        else:
            while count < n and x[count] <= x_eval[i]:
                count += 1
            counts[i] = count

    for orddo in range(0, order + 1):
        coefs[0] = 1
        coefs[orddo] = 1
        if orddo > 1:
            num = 1.
            for j in range(2, orddo + 1):
                num *= j
            denom1 = 1.
            denom2 = num / orddo
            for i in range(2, orddo + 1):
                coefs[i - 1] = num / (denom1 * denom2)
                denom1 *= i
                denom2 /= orddo - i + 1
        denom = np.power(h, orddo)
        
        ix = 0
        for i in range(n_eval):
            ix = np.round(counts[i])
            if ix == 0:
                exp_mult = max(np.exp((x_eval[i] - x[0]) / h), K_EPS)
                output[i] += (
                    betas[orddo] * np.power(x[0] - x_eval[i], orddo) 
                    / denom * exp_mult * y[0]
                )
                for j in range(orddo + 1):
                    output[i] += (
                        betas[orddo] * coefs[j] * np.power(-x_eval[i], orddo - j)
                        * Ry[j, 0] / denom * exp_mult
                    )
            else:
                exp_mult = np.exp((x[ix - 1] - x_eval[i]) / h)
                for j in range(orddo + 1):
                    output[i] += betas[orddo] * coefs[j] * (
                        np.power(x_eval[i], orddo - j) * Ly[j, ix - 1] * exp_mult
                        + np.power(-x_eval[i], orddo - j) * Ry[j, ix - 1] / exp_mult
                    ) / denom


def kernel_eval(x_eval, order=4):
    """Evaluates the polynomial-exponential kernel."""    
    betas = betas_for_order(order)
    h = h_Gauss_to_K(1., betas)
    x_eval = np.sort(x_eval)

    output = np.zeros_like(x_eval)
    counts = np.zeros_like(x_eval).astype(np.int64)
    coefs = np.zeros_like(betas)
    Ly = np.zeros((order + 1, 1), order="C")
    Ry = np.zeros((order + 1, 1), order="C")

    ksum_numba(
        x=np.array([0.]),
        y=np.array([1.]),
        x_eval=x_eval,
        h=h,
        betas=betas,
        output=output,
        counts=counts,
        coefs=coefs,
        Ly=Ly,
        Ry=Ly,
    )
    pdf = output /  h
    pdf[np.isnan(pdf)] = K_EPS
    pdf[~np.isfinite(pdf)] = K_EPS

    return pdf


def kernel_density(x, x_eval, weights=None, bw_method="silverman", order=4):
    """Kernel density estimation with poly-exponential kernel.

    See https://github.com/DavidHofmeyr/FKSUM and
    "Fast exact evaluation of univariate kernel sums" (Hofmeyr, 2019).

    Parameters
    ----------
    x : ndarray of shape (n,)
        Training data (kernel locations). Usually the observed sample, 
        or bin locations for binned summing. 

    x_eval : ndarray of shape (n_eval,)
        Points at which to evaluate the sums. Should be sorted in
        non-decreasing order.

    weights : optional ndarray of shape (n,)
        Weights of datapoints. This must be the same shape as x. 
        If None (default), the samples are assumed to be equally weighted.

    bw_method : str or scalar
        The method used to calculate the estimator bandwidth. This can be 
        ‘silverman’ or a scalar constant or a callable. If a scalar, 
        this will be multiplied with the data standard deviation, as in
        scipy.stats.gaussian_kde.

    order : int
        The order of the polynomial for the polynomial-exponential kernel.

    Returns
    -------
    pdf : ndarray of shape(n_eval,)
        Evaluated kernel density at x_eval.
    """

    if weights is None:
        weights = np.ones_like(x)
    n, = x.shape
    n, = weights.shape
    n_eval, = x_eval.shape

    sort_ixs = np.argsort(x)
    x = x[sort_ixs]
    weights = weights[sort_ixs]
    x_eval = np.sort(x_eval)

    betas = betas_for_order(order)

    if bw_method == "silverman":
        h = silverman(x, betas)
    else:
        h_Gauss = bw_method * np.std(x)
        h = h_Gauss_to_K(h_Gauss, betas)

    output = np.zeros_like(x_eval)
    counts = np.zeros_like(x_eval).astype(np.int64)
    coefs = np.zeros_like(betas)
    Ly = np.zeros((order + 1, n), order="C")
    Ry = np.zeros((order + 1, n), order="C")

    mean = np.mean(x)
    ksum_numba(
        x=x-mean,
        y=weights,
        x_eval=x_eval-mean,
        h=h,
        betas=betas,
        output=output,
        counts=counts,
        coefs=coefs,
        Ly=Ly,
        Ry=Ry,
    )
    pdf = output / n / h
    pdf[np.isnan(pdf)] = K_EPS
    pdf[~np.isfinite(pdf)] = K_EPS

    # XXX - need to unsort pdf
    return pdf
