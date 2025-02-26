from scipy.stats import norm
import numpy as np

def UCB(x, model, ucb_kappa):
    mean, std = model.predict(x.reshape(1, -1), return_std=True)
    return mean + ucb_kappa * std

def EI(x, model, best_f, xi=0.01):
    """
    Expected Improvement (EI) acquisition function.

    Parameters:
    x : array-like
        Input point(s) to evaluate the acquisition function.
    model : GaussianProcessRegressor
        Trained GP model to predict mean and standard deviation.
    best_f : float
        Best observed objective function value.
    xi : float, optional (default=0.01)
        Exploration-exploitation tradeoff parameter.

    Returns:
    float
        EI value at x.
    """
    mean, std = model.predict(x.reshape(1, -1), return_std=True)
    std = std + 1e-9  # Avoid division by zero
    improvement = mean - best_f - xi
    Z = improvement / std
    return improvement * norm.cdf(Z) + std * norm.pdf(Z)


def PI(x, model, best_f, xi=0.01):
    """
    Probability of Improvement (PI) acquisition function.

    Parameters:
    x : array-like
        Input point(s) to evaluate the acquisition function.
    model : GaussianProcessRegressor
        Trained GP model to predict mean and standard deviation.
    best_f : float
        Best observed objective function value.
    xi : float, optional (default=0.01)
        Exploration-exploitation tradeoff parameter.

    Returns:
    float
        PI value at x.
    """
    mean, std = model.predict(x.reshape(1, -1), return_std=True)
    std = std + 1e-9  # Avoid division by zero
    Z = (mean - best_f - xi) / std
    return norm.cdf(Z)
