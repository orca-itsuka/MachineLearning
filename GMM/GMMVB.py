import csv
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from numpy import linalg as la
from scipy.special import digamma, gamma, logsumexp
from mpl_toolkits.mplot3d import Axes3D
import sys

class GMM_VB():
    def __init__(self, K=6, alpha0=0.1):
        self.K = K
        self.alpha0 = alpha0

    def init_params(self, X):
        self.N, self.D = X.shape
        self.m0 = np.random.randn(self.D)
        self.beta0 = np.array([1.0])
        self.W0 = np.eye(self.D)
        self.nu0 = np.array([self.D])

        self.N_k = (self.N / self.K) + np.zeros(self.K)

        self.alpha = np.ones(self.K) * self.alpha0
        self.beta = np.ones(self.K) * self.beta0
        self.m = np.random.randn(self.K, self.D)
        self.W = np.tile(self.W0, (self.K, 1, 1))
        self.nu = np.ones(self.K) * self.D

        self.Mu = self.m
        self.Sigma = np.zeros((self.K, self.D, self.D))
        for k in range(self.K):
            self.Sigma[k] = la.inv(self.nu[k] * self.W[k])
