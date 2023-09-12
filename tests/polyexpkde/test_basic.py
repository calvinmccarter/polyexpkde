import numpy as np
import pytest
import scipy.stats as spst

from polyexpkde import PolyExpKernelDensity


def test_gaussian_approx():
    rng = np.random.default_rng(12345)
    X = rng.lognormal(size=(1000,))
    X_eval = np.linspace(-5, 15, 10000)

    gkde = spst.gaussian_kde(X, bw_method="silverman")
    gpdf = gkde.pdf(X_eval)
    gsample = gkde.resample(20, seed=12345)

    fkde = PolyExpKernelDensity(bw_method="silverman").fit(X[:, None])
    fpdf = fkde.pdf(X_eval[:, None])
    fsample = fkde.resample(20, seed=12345)

    assert (gpdf.shape[0], 1) == fpdf.shape
    np.testing.assert_allclose(gpdf, fpdf.flatten(), rtol=0., atol=0.005)
    
    assert gsample.shape == fsample.shape
    res = spst.ks_2samp(gsample.flatten(), fsample.flatten())
    assert res.pvalue > 0.03
