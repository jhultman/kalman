"""
TODO: Tune process and measurement covariance.

Notation:

    A: n x n, state transition
    H: m x n, measurement viewpoint
    K: n x m, kalman gain
    P: n x n, error covariance
    R: m x m, process noise covariance
    Q: n x n, measurement noise covariance

    x: n x 1, state
    w: n x 1, process noise
    v: n x 1, measurement noise
"""


import numpy as np
from numpy.linalg import multi_dot
from numpy.random import multivariate_normal
np.random.seed(21)


class LinearSystem:
    
    def __init__(self, n, m, x0, A, H, R, Q):
        self.n = n
        self.m = m
        self.x = x0
        self.A = A
        self.H = H
        self.R = R
        self.Q = Q
        
    def forward(self):
        state_transition = self.A @ self.x
        process_noise = multivariate_normal(np.zeros(self.n), self.Q)
        self.x = state_transition + process_noise
        
    def measurement(self):
        measurement_noise = multivariate_normal(np.zeros(self.m), self.R)
        self.z = self.H @ self.x + measurement_noise


class Kalman:
    
    def __init__(self, n, m, sys):
        self.linear_system = sys
        self.n = n
        self.m = m
        self.K = np.zeros((n, m))
        self.P = np.eye(n)
        self.x_est = np.zeros(n)
        
    def update_kalman_gain(self):
        sys = self.linear_system
        prod = multi_dot((sys.H, self.P, sys.H.T))
        inv = np.linalg.pinv(prod + sys.R)
        self.K = multi_dot((self.P, sys.H.T, inv))

    def update_estimate(self):
        sys = self.linear_system
        innovation = sys.z - sys.H @ self.x_est
        self.x_est = self.x_est + self.K @ innovation
    
    def update_covariance(self):
        H = self.linear_system.H
        I = np.eye(self.n)
        self.P = (I - self.K @ H) @ self.P
        
    def project_ahead(self):
        sys = self.linear_system
        self.x_est = sys.A @ self.x_est
        self.P = multi_dot((sys.A, self.P, sys.A.T)) + sys.Q
        
    def loop(self):
        self.linear_system.forward()
        self.linear_system.measurement()
        
        self.update_kalman_gain()
        self.update_estimate()
        self.update_covariance()
        self.project_ahead()
