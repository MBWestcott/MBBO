import numpy as np
from scipy import optimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor

class AcquisitionFunction:
    def __init__(self, x, model, best_f=None, xi=0.1, kappa=0.8):
        """
        Base class for acquisition functions.

        Parameters:
        -----------
        model : object
            A surrogate model with a predict method (e.g., GaussianProcessRegressor).
        x : array-like
            The input point(s) at which to evaluate the acquisition function.
        best_f : float, optional
            The best observed function value (used for EI and PI).
        xi : float, optional
            Exploration-exploitation tradeoff parameter.
        kappa : float, optional
            For UCB - weights the standard deviation towards exploration.
        """
        self.model = model
        self.x = np.array(x)
        self.best_f = best_f
        self.xi = xi
        self.kappa = kappa

    def apply(self, x):
        """
        Evaluate the acquisition function at x using self.model.
        This method should be overridden by subclasses.
        
        Parameters:
        -----------
        x : array-like
            The input point(s) to evaluate the acquisition function.
        
        Returns:
        --------
        float
            The computed acquisition function value.
        """
        raise NotImplementedError("Subclasses must implement the apply() method.")
    
    def maximize(self, initial_x, bounds):
        return optimize.minimize( lambda arg: -self.apply(arg), x0=initial_x, bounds=bounds)

    def do_predict(self, model, x):
        if isinstance(model, GaussianProcessRegressor):
            mean, std = model.predict(x.reshape(1, -1), return_std=True)
            return mean, std
        elif isinstance(self.model, RandomForestRegressor):
            # doesn't natively return std so need to get it from individual estimators
            tree_predictions = np.array([tree.predict(x.reshape(1, -1)) for tree in self.model.estimators_])
            mean = tree_predictions.mean(axis=0)
            std = tree_predictions.std(axis=0)
            return mean, std
        else:
            raise ValueError("Model must be either GaussianProcessRegressor or RandomForestRegressor")
        
class UCB(AcquisitionFunction):
    def apply(self, x):
        """
        Upper Confidence Bound (UCB) acquisition function.
        """
        mean, std = self.do_predict(self.model, x)
        return mean + self.kappa * std        
        
class EI(AcquisitionFunction):
    def apply(self, x):
        """
        Expected Improvement (EI) acquisition function.
        Uses self.best_f as the current best observation and self.xi for exploration.
        """
        mean, std = self.do_predict(self.model, x)
        std = std + 1e-9  # Avoid division by zero
        improvement = mean - self.best_f - self.xi
        Z = improvement / std
        return improvement * norm.cdf(Z) + std * norm.pdf(Z)

class PI(AcquisitionFunction):
    def apply(self, x):
        """
        Probability of Improvement (PI) acquisition function.
        Uses self.best_f as the current best observation and self.xi for exploration.
        """
        mean, std = self.do_predict(self.model, x)
        std = std + 1e-9  # Avoid division by zero
        Z = (mean - self.best_f - self.xi) / std
        return norm.cdf(Z)