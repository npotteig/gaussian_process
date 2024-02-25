import numpy as np
import scipy as sp

from kernel import Kernel

class GaussianProcessModelSelection:
    
    kernel_type = "rbf"
    
    @classmethod
    def _nll_fn(cls, X_train, Y_train, noise):
        """
        Returns a function that computes the negative log marginal
        likelihood for training data X_train and Y_train and given noise level.
        
        Args:
            X_train: training locations (m x d).
            Y_train: training targets (m x 1).
            noise: known noise level of Y_train.
        
        Returns:
            Minimization objective.
        """
        
        Y_train = Y_train.ravel()
        
        def step(theta):
            # Numerically more stable implementation of log marginal likelihood as described
            # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
            # 2.2, Algorithm 2.1.
            kernel = Kernel.kernel(cls.kernel_type, l=theta[0], sigma_f=theta[1])
            K = kernel(X_train, X_train) + \
                noise**2 * np.eye(len(X_train))
            L = np.linalg.cholesky(K)
            
            S1 = sp.linalg.solve_triangular(L, Y_train, lower=True)
            S2 = sp.linalg.solve_triangular(L.T, S1, lower=False)
            
            return np.sum(np.log(np.diagonal(L))) + \
                0.5 * Y_train.dot(S2) + \
                0.5 * len(X_train) * np.log(2*np.pi)
        
        return step
    
    @classmethod
    def optimize_hyperparameters(cls, X_train, Y_train, noise, n_restarts=10):
        """
        Optimizes the hyperparameters of the kernel function.
        
        Args:
            X_train: training locations (m x d).
            Y_train: training targets (m x 1).
            noise: known noise level of Y_train.
            n_restarts: number of restarts of the local optimizer.
        
        Returns:
            Best hyperparameters.
        """
        
        l = np.random.uniform(0, 10, n_restarts)
        sigma_f = np.random.uniform(0, 1, n_restarts)
        best_params = {}
        best_nll = np.inf
        
        for i in range(n_restarts):
            theta_initial = [l[i], sigma_f[i]]
            res = sp.optimize.minimize(cls._nll_fn(X_train, Y_train, noise), 
                                       theta_initial, 
                                       bounds=((1e-5, None), (1e-5, None)),
                                       method='L-BFGS-B')
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = {'l': res.x[0], 'sigma_f': res.x[1]}
        
        return best_params
    

if __name__ == '__main__':
    from gp_utils import plot_gp
    from gp_regression import GaussianProcessRegression
    import matplotlib.pyplot as plt
    
    noise = 0.4

    # Noisy training data
    X_train = np.arange(-3, 4, 1).reshape(-1, 1)
    Y_train = np.sin(X_train) + noise * np.random.randn(*X_train.shape)
    
    X = np.arange(-5, 5, 0.2).reshape(-1, 1)
    
    best_hyperparameters = GaussianProcessModelSelection.optimize_hyperparameters(X_train, Y_train, noise)
    gp = GaussianProcessRegression(**best_hyperparameters)
    mu_s, cov_s = gp.compute_posterior(X, X_train, Y_train)
    
    samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
    plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples)
    plt.show()