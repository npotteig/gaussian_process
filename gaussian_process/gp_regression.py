import numpy as np

from kernel import Kernel

class GaussianProcessRegression:
    def __init__(self, kernel_type="rbf", sigma_y=1e-8, *args, **kwargs):
        self.kernel = Kernel.kernel(kernel_type, **kwargs)
        self.sigma_y = sigma_y
        self.X_train = None
        self.y_train = None
        self.K = None
        self.K_inv = None
        self.K_s = None
        self.K_ss = None
    
    def compute_posterior(self, X_s, X_train, y_train):
        """
        Computes the suffifient statistics of the posterior distribution 
        from m training data X_train and Y_train and n new inputs X_s.
        
        Utilizes:
            sigma_y: Noise parameter.
        
        Args:
            X_s: New input locations (n x d).
            X_train: Training locations (m x d).
            Y_train: Training targets (m x 1).
        
        Returns:
            Posterior mean vector (n x d) and covariance matrix (n x n).
        """
        
        self.X_train = X_train
        self.y_train = y_train
        
        self.K = self.kernel(X_train, X_train) + self.sigma_y**2 * np.eye(len(X_train))
        self.K_inv = np.linalg.inv(self.K)
        self.K_s = self.kernel(X_train, X_s)
        self.K_ss = self.kernel(X_s, X_s) + self.sigma_y**2 * np.eye(len(X_s))
        
        mu_s = self.K_s.T.dot(self.K_inv).dot(y_train)
        
        cov_s = self.K_ss - self.K_s.T.dot(self.K_inv).dot(self.K_s)
        
        return mu_s, cov_s
    
if __name__ == '__main__':
    from gp_utils import plot_gp
    import matplotlib.pyplot as plt
    gp = GaussianProcessRegression()
    
    # Noise free training data
    X_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
    Y_train = np.sin(X_train)
    
    X = np.arange(-5, 5, 0.2).reshape(-1, 1)

    # Compute mean and covariance of the posterior distribution
    mu_s, cov_s = gp.compute_posterior(X, X_train, Y_train)

    samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
    plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples)
    plt.show()