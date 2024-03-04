from typing import Tuple
import numpy as np

from kernel import Kernel

# Class for computing the Laplace Approximation, and performing binary Gaussian Process classification
class GaussianProcessClassifier:
    _KERNEL_TYPE = 'rbf'
    
    # --- Constructor, should take in:
    #   (1) features (X) -> n x d matrix
    #   (2) class labels (y) -> vector of length n, containing 0 or 1
    #   (3) hyperparameters for the Kernel
    # ----> Within the constructor, you should create the Gaussian process for the data, and store it as an instance variable - will be used throughout most methods
    def __init__(self, kernel_params: dict):
        self.kernel = Kernel.kernel(self._KERNEL_TYPE, **kernel_params)
    #

    # --- compute the negative log likelihood for latent function f: this is not necessary, but a very useful debugging tool for your optimization method
    def _NLL(self,f, y):
        return -np.sum(y * np.log(self._sigmoid(f)) + (1-y) * np.log(self._sigmoid(-f)))
    #

    # --- compute and return vectorized application of the sigmoid (logistic function) to the given latent function values f
    # ----> Assumption: f is an arbitrary numpy array
    def _sigmoid(self, f: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-f))

    # --- compute and return the numerically-stable inversion of the Hessian, via application of the Woodbury matrix inversion lemma
    # ----> K is the GP covariance matrix, and dd_sigmoid is a vector of all second-order partial derivatives of the log likelihood
    def _woodbury_inversion(self, K, dd_sigmoid, n):
        sqrt_dd_sigmoid = np.zeros((n, n))
        # Compute square root of W matrix
        np.fill_diagonal(sqrt_dd_sigmoid, np.sqrt(dd_sigmoid))

        H_inv = K - K @ sqrt_dd_sigmoid @ np.linalg.inv(np.eye(n) + sqrt_dd_sigmoid @ K @ sqrt_dd_sigmoid) @ sqrt_dd_sigmoid @ K
        return H_inv

    # --- Newton's method for computing the MAP
    # ----> recommended to store the mode as an instance variable, for later use
    def _newton_MAP(self, X, y, n_steps=30):
        # --- recommended initialization for the mode
        n, _ = X.shape
        mode = 1e-1*np.random.rand(n)
        H_l = np.zeros((n, n))

        for i in range(n_steps):
            sig_f = self._sigmoid(mode)
            dd_sigmoid = sig_f * (1 - sig_f)
            np.fill_diagonal(H_l, dd_sigmoid)
            # Take one newton step t -> t+1
            mode = self._woodbury_inversion(self.kernel(X, X), dd_sigmoid, n) @ (H_l @ mode + y - sig_f)
            print("At newton step:",i+1,"the neg-log-likelihood is:", self._NLL(mode, y))
        return mode

    # --- compute the predictive distribution for the latent function - for each point in X_star, this should be the predictive mean, and predictive variance
    def _latent_predictive_distribution(self, X, y, X_star, mode):
        n, _ = X.shape
        m, _ = X_star.shape
        H_l = np.zeros((n, n))
        
        sig_f = self._sigmoid(mode)
        W = sig_f * (1 - sig_f)
        np.fill_diagonal(H_l, W)

        k_star = self.kernel(X, X_star)
        inv_term = np.linalg.inv(np.linalg.inv(H_l) + self.kernel(X, X))
        sub_term = k_star.T @ inv_term @ k_star

        variances = Kernel.get_param("sigma_f", 1.0) - np.diag(sub_term)

        means = k_star.T @ (y - sig_f)
        return means, variances

    # --- compute the averaged predictive probability, given a set of predictive distributions (a mean and variance for each example)
    # ----> this should be done via Monte Carlo integration, with `k` being the number of samples (feel free to adjust)
    def _averaged_predictive_probability(self,predictive_mean,predictive_variance,k=5000):
        # Sample k samples from predictive distribution and take there mean for each point
        return self._sigmoid(np.random.normal(predictive_mean, predictive_variance, size=(k, predictive_mean.shape[0])).T).mean(axis=1)
    
    def train(self, X: np.ndarray, y: np.ndarray, n_steps: int=30) -> np.ndarray:
        return self._newton_MAP(X, y, n_steps=n_steps)
    
    def predict(self, X: np.ndarray, y: np.ndarray, X_star: np.ndarray, mode: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._latent_predictive_distribution(X, y, X_star, mode)
    
    def compute_probs(self, X_pred_mean: np.ndarray, X_pred_var: np.ndarray) -> np.ndarray:
        return self._averaged_predictive_probability(X_pred_mean, X_pred_var)
    
    # --- compute classification accuracy for GP classifier probabilities (gp_probs), given ground truth labels (gt_labels)
    def classification_accuracy(self, gp_probs: np.ndarray, gt_labels: np.ndarray) -> float:
        # Predict all data points with probability > 0.5 as 1, o/w 0
        preds = np.where(gp_probs > 0.5, 1, 0)
        return np.count_nonzero(preds == gt_labels) / gp_probs.shape[0]

if __name__=='__main__':
    import json
    # dataset name
    glue_prefix = 'cola'

    # training data: inputs, targets, and sentences
    X_train = np.load('data/'+glue_prefix+'_X_train.npy')
    y_train = np.load('data/'+glue_prefix+'_y_train.npy')
    sentences_train = json.load(open('data/'+glue_prefix+'_sentences_train.json','r'))

    # validation data: inputs, targets, and sentences
    X_val = np.load('data/'+glue_prefix+'_X_val.npy')
    y_val = np.load('data/'+glue_prefix+'_y_val.npy')
    sentences_val = json.load(open('data/'+glue_prefix+'_sentences_val.json','r'))
    sentences_val = np.array(sentences_val)

    # hyperparameters to consider, as part of your analysis
    all_hyperparams = []
    all_hyperparams.append({'sigma_f':10,'l':.7})
    all_hyperparams.append({'sigma_f':10,'l':1})
    all_hyperparams.append({'sigma_f':10,'l':2})
    all_hyperparams.append({'sigma_f':10,'l':2})
    all_hyperparams.append({'sigma_f':5,'l':2})
    all_hyperparams.append({'sigma_f': 10, 'l': 3})
    all_hyperparams.append({'sigma_f':10,'l':4})

    max_val_accuracy = 0
    max_val_classifier = {}
    for i in range(len(all_hyperparams)):
        g_classifier = GaussianProcessClassifier(all_hyperparams[i])
        mode = g_classifier.train(X_train, y_train, n_steps=10)
        
        train_pred_mean, train_pred_var = g_classifier.predict(X_train, y_train, X_train, mode)
        val_pred_mean, val_pred_var = g_classifier.predict(X_train, y_train, X_val, mode)

        train_g_probs = g_classifier.compute_probs(train_pred_mean, train_pred_var)
        val_g_probs = g_classifier.compute_probs(val_pred_mean, val_pred_var)

        train_accuracy = g_classifier.classification_accuracy(train_g_probs, y_train)
        val_accuracy = g_classifier.classification_accuracy(val_g_probs, y_val)
        print("Accuracy for signal_variance:", all_hyperparams[i]['sigma_f'], "l", all_hyperparams[i]['l'],
              "is: Training:",train_accuracy,"Validation:", val_accuracy)

        # Select model that gives the highest validation accuracy
        if val_accuracy > max_val_accuracy:
            max_val_classifier = {'classifier': g_classifier, 'probs': val_g_probs}
