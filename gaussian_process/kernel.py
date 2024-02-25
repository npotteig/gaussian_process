import numpy as np

class Kernel:

    rbf_kernel = "rbf"
    linear_kernel = "linear"
    hyperparameters = {}

    @classmethod
    def kernel(cls, kernel_type: str, **kwargs):
        cls.hyperparameters = kwargs
        if kernel_type == cls.rbf_kernel:
            return cls.rbf
        elif kernel_type == cls.linear_kernel:
            return cls.linear
        else:
            raise ValueError("Invalid kernel type")
    
    @classmethod
    def rbf(cls, x1, x2):
        """
        Isotropic squared exponential kernel.
        
        Utilizes: 
            l: Kernel length parameter.
            sigma_f: Kernel vertical variation parameter.
        
        Args:
            X1: Array of m points (m x d).
            X2: Array of n points (n x d).

        Returns:
            (m x n) matrix.
        """
        
        sigma_f = cls.hyperparameters.get("sigma_f", 1.0)
        l = cls.hyperparameters.get("l", 1.0)
        sq_dist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return sigma_f * np.exp(-0.5 / l**2 * sq_dist)
    
    @classmethod
    def linear(cls, X, Y):
        return X @ Y.T 