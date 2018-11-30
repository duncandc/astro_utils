"""
schechter function classes
"""

import numpy as np
from warnings import warn
try:
    from mpmath import gammainc
    no_mpmath = False
except ImportError:
    from scipy.special import gammainc
    no_mpmath = True

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

    def number_density(self, a, b):
        """
        Intgrate the Schechter function over the bounds [a,b].
        
        Parameters
        ----------
        a : float
            faint limit

        b : float
            bright limit

        Returns
        -------
        N : numpy.array

        Notes
        -----
        """

        if no_mpmath & (self.alpha <= 0):
            msg = ('mpmath packlage must be installed in order',
                   'to perform this calculation for alpha<=0.')
            raise ValueError(msg)
        
        if not isinstance(a, float):
            msg = ('`a` argument must be a float.')
            raise ValueError(msg)
        if not isinstance(b, float):
            msg = ('`b` argument must be a float.')
            raise ValueError(msg)

        a = a / self.x0
        b = b / self.x0

        l = float(gammainc(self.alpha + 1, a))
        r = float(gammainc(self.alpha + 1, b))
        return (l - r)*self.phi0


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
        val = norm*(10.0**(0.4*(self.M0-m)))**(self.alpha+1.0)*np.exp(-10.0**(0.4*(self.M0-m)))
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


    def number_density(self, a, b):
        """
        Intgrate the Schechter function over the bounds [a,b].
        
        Parameters
        ----------
        a : float
            faint limit

        b : float
            bright limit

        Returns
        -------
        N : numpy.array

        Notes
        -----
        """

        if no_mpmath & (self.alpha <= 0):
            msg = ('mpmath packlage must be installed in order',
                   'to perform this calculation for alpha<=0.')
            raise ValueError(msg)
        
        if not isinstance(a, float):
            msg = ('`a` argument must be a float.')
            raise ValueError(msg)
        if not isinstance(b, float):
            msg = ('`b` argument must be a float.')
            raise ValueError(msg)

        x0 = 10**(-0.4*self.M0)
        a = 10**(-0.4*a) / x0
        b = 10**(-0.4*b) / x0
        l = float(gammainc(self.alpha + 1, a))
        r = float(gammainc(self.alpha + 1, b))
        return (l - r)*self.phi0


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

    def __call__(self, x):
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

    def number_density(self, a, b):
        """
        Intgrate the Schechter function over the bounds [a,b].
        
        Parameters
        ----------
        a : float
            faint limit

        b : float
            bright limit

        Returns
        -------
        N : numpy.array

        Notes
        -----
        """

        if no_mpmath & (self.alpha <= 0):
            msg = ('mpmath packlage must be installed in order',
                   'to perform this calculation for alpha<=0.')
            raise ValueError(msg)
        
        if not isinstance(a, float):
            msg = ('`a` argument must be a float.')
            raise ValueError(msg)
        if not isinstance(b, float):
            msg = ('`b` argument must be a float.')
            raise ValueError(msg)

        x0 = 10**(self.x0)
        a = 10**(a) / x0
        b = 10**(b) / x0
        l = float(gammainc(self.alpha + 1, a))
        r = float(gammainc(self.alpha + 1, b))
        return (l - r)*self.phi0


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

