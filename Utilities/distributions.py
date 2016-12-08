import numpy as np
import scipy.stats as stats
import scipy
import statsmodels.api as sm
from scipy.interpolate import interp1d

def gauss(x, sig, mu=0):
    """
    Return x from a gaussian distribution

    :param x: Array like
    :param mu: mean of normal distribution
    :param sig: standard deviation of normal distribution
    """

    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def tophat(x, delta):
    """
    Return x from a gaussian distribution

    :param x: Array like
    :param delta: value < delta set to 1
    """
    ret = np.zeros_like(x)
    ret[np.abs(x) < delta] = 1
    return ret

def convolved(x,sig,broadfac=2):
    delta = 1e-4
    big_grid = np.arange(-10*sig,10*sig,delta*sig)
    # Cannot analytically convolve continuous PDFs, in general.
    # So we now make a probability mass function on a fine grid 
    # - a discrete approximation to the PDF, amenable to FFT...
    simple = stats.uniform(loc=-sig*broadfac,scale=2*sig*broadfac)
    err1 = stats.norm(loc=0,scale=sig)
    err = stats.norm(loc=0,scale=sig/broadfac)
    pmf1 = simple.pdf(big_grid)*delta
    pmf2 = err.pdf(big_grid)*delta
    pmf3 = err1.pdf(big_grid)*delta
    conv_pmf = scipy.signal.fftconvolve(pmf1,pmf2,'same') # Convolved probability mass function
    conv_pmf = conv_pmf/sum(conv_pmf)
    f = interp1d(big_grid,conv_pmf/max(conv_pmf))
    g = interp1d(big_grid,pmf1/max(pmf1))
    h = interp1d(big_grid,pmf3/max(pmf3))
    return f(x),g(x),h(x)

