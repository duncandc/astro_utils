"""
schechter function classes
"""

import numpy as np
from warnings import warn

__all__=['Schechter', 'MagSchechter', 'LogSchechter']

class Schechter():
    """
    Schechter function class
    """
    def __init__(self, phi0, x0, alpha):
        """
        """
        self.phi0 = phi0
        self.x0 = x0
        self.alpha = alpha

    def __call__(self, x):
        """
        """
        x = np.atleast_1d(x)

        norm = self.phi0/self.x0
        val = norm * (x/self.x0)**self.alpha * np.exp(-x)

        return val

    def rvs(self, x_min, size=100, max_iter=100):
        """
        Parameters
        ----------
        size : int
            number of random variates to return

        max_iter : int
            maximum number of iterations to preform when calculating random variates

        Returns
        -------
        x : numpy.array
            array of random variates sampled from the Schechter function

        Notes
        -----
        """
        return _sample_schechter(self.x0, self.alpha, x_min, size=size, max_iter=max_iter)


class MagSchechter(object):
    """
    Magnitudes Schechter function class
    """
    def __init__(self, phi0, M0, alpha):
        """
        """
        self.phi0 = phi0
        self.M0 = M0
        self.alpha = alpha

    def __call__(self, m):
        """
        """
        m = np.atleast_1d(m)

        norm = (2.0/5.0)*self.phi0*np.log(10.0)
        val = norm*(10.0**(0.4*(self.M0-x)))**(self.alpha+1.0)*np.exp(-10.0**(0.4*(self.M0-x)))
        return val

    def rvs(self, m_max, size=100, max_iter=100):
        """
        Parameters
        ----------
        size : int
            number of random variates to return

        max_iter : int
            maximum number of iterations to preform when calculating random variates

        Returns
        -------
        x : numpy.array
            array of random variates sampled from the Schechter function

        Notes
        -----
        """

        x_min = 10**(-0.4*m_max)
        x0 = 10**(-0.4*self.M0)

        x = _sample_schechter(x0, self.alpha, x_min, size=size, max_iter=max_iter)
        return -2.5*np.log10(x)


class LogSchechter():
    """
    Log Schecter function class
    """
    def __init__(self, phi0, x0, alpha):
        """
        """
        self.phi0 = phi0
        self.x0 = x0
        self.alpha = alpha

    def __call__(self, m):
        """
        """
        x = np.atleast_1d(x)
        norm = np.log(10.0)*self.phi0
        val = norm*(10.0**((x-self.x0)*(1.0+self.alpha)))*np.exp(-10.0**(x-self.x0))
        return val

    def rvs(self, x_min, size=100, max_iter=100):
        """
        Parameters
        ----------
        size : int
            number of random variates to return

        max_iter : int
            maximum number of iterations to preform when calculating random variates

        Returns
        -------
        x : numpy.array
            array of random variates sampled from the Schechter function

        Notes
        -----
        """

        x_min = 10**(x_min)
        x0 = 10**(self.x0)

        x = _sample_schechter(x0, self.alpha, x_min, size=size, max_iter=max_iter)
        return np.log10(x)


def _sample_schechter(x0, alpha, x_min, size=100, max_iter=1000):
        """
        Return random samples from a schechter function.
        This method follows Richard Gill's method given here:
        http://www.math.leidenuniv.nl/~gill/teaching/astro/stanSchechter.pdf

        Parameters
        ----------
        size : int
            number of random variates to return

        max_iter : int
            maximum number of iterations to preform when calculating random variates

        Returns
        -------
        x : numpy.array
            array of random variates sampled from the Schechter function

        Notes
        -----
        """
        out = []
        n = 0
        num_iter = 0
        while (n<size) & (num_iter<max_iter):
            x = np.random.gamma(scale=x0, shape=alpha+2, size=size)
            x = x[x>x_min]
            u = np.random.uniform(size=x.size)
            x = x[u<x_min/x]
            out.append(x)
            n+=x.size
            num_iter += 1

        if num_iter >= max_iter:
            msg = ("The maximum number of iterations reached.",
                   "Random variates may not be representitive.",
                   "Try increasing `max_iter`.")
            print(msg)

        return np.concatenate(out)[:size]

