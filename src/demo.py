import numpy as np
import matplotlib.pyplot as plt
from kalman import Kalman, LinearSystem

np.random.seed(21)


def run_filter(kalman, num_iters=100):
    x = []
    x_est = []
     
    for _ in range(num_iters):
        x.append(kalman.linear_system.x)
        x_est.append(kalman.x_est)
        kalman.loop()

    x = np.array(x)
    x_est = np.array(x_est)
    return x, x_est


def plot_kalman(x, x_est):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    for a, xi, xi_est in zip(ax, x.T, x_est.T):
        a.plot(xi, 'blue')
        a.plot(xi_est, 'red')
        a.set_xlabel('Iteration')
        a.set_ylabel('State')
        a.legend(['True', 'Predicted'])
        
    fig.suptitle('Kalman filter')
    plt.savefig('../images/kalman.png')


def init_demo_kalman():
    n = m = 2

    A = np.array([[0.9, 0.2], [-0.3, 0.5]])
    H = np.array([[1.0, 0.1], [-0.2, 1.1]])

    R = 0.5 * np.eye(m)
    Q = 1.4 * np.eye(n)

    x0 = [-0.7, 0.9]

    sys = LinearSystem(n, m, x0, A, H, R, Q)
    kalman = Kalman(n, m, sys)
    return kalman


def main():
    kalman = init_demo_kalman()
    x, x_est = run_filter(kalman)
    plot_kalman(x, x_est)


if __name__ == '__main__':
    main()
