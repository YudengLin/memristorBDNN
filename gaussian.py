from __future__ import division, print_function
import numpy as np

class gaussian:
    def __init__(self, mu, var):
        self.mu = mu
        self.var = var
        self.sigma = np.sqrt(var)

    def loglik(self, weights):
        exponent = -0.5 * (weights - self.mu) ** 2 / self.var
        log_coeff = -0.5 * (np.log(2 * np.pi) + 2 * np.log(self.sigma))

        return (exponent + log_coeff).sum()
