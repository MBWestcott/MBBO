import numpy as np

# Result of running a test on a single test function
class FunctionTestResult():
    def __init__(self, iterations, test_function_number):
        self.iterations = iterations
        self.test_function_number = test_function_number
      

class FunctionTestResults():
    results = []
    def __init__(self, competition_function_number, params_description):
        self.competition_function_number = competition_function_number
        self.params_description = params_description

    def append(result:FunctionTestResult):
        results.append(result)


#Test bimodal function
class one_d_test:
    # Parameters for the two Gaussian peaks
    #mu1 is x of first peak
    #mu2 is x of second peak TODO: experiment with 0.02-0.2
    #Sigmas are widths of peaks
    #alpha1 is weight of first peak (default 1)
    #alpha2 is weight of second peak (default 0.5)

    sigma1 = 0.1
    sigma2 = 0.1
    mu1 = 0.3
    mu2 = 0.7
    alpha1 = 1
    alpha2 = 0.5

    def __init__(self, sigma1=0.1, sigma2=0.1, mu1=0.3, mu2=0.7, alpha1=1, alpha2=0.5):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.mu1 = mu1
        self.mu2 = mu2
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def call_function(self, x): 
        # Calculate the two Gaussian components
        gaussian1 = np.exp(-((x - self.mu1) ** 2) / (2 * self.sigma1 ** 2))
        gaussian2 = np.exp(-((x - self.mu2) ** 2) / (2 * self.sigma2 ** 2))
        gaussian1 = self.alpha1 * gaussian1
        gaussian2 = self.alpha2 * gaussian2
        
        # Combine and normalize to keep the output between 0 and 1
        result = (gaussian1 + gaussian2) / (self.alpha1 + self.alpha2)
        return result
    

class two_d_test:

    def __init__(self, sigma1=0.1, sigma2=0.1, mu1=[0.3,0.3], mu2=[0.7,0.7], alpha1=1, alpha2=0.5):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.mu1 = mu1
        self.mu2 = mu2
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    sigma1 = 0.1
    sigma2 = 0.1
    mu1 = [0.3,0.3]
    mu2 = [0.7,0.7]
    alpha1 = 1
    alpha2 = 0.5

    def call_function(self, X):
        """
        Bimodal function f(x0, x1) in [0, 1]^2 -> [0, 1].
        X is an array of shape (N, 2).
        Returns a 1D array of length N, each element in [0, 1].
        """

        X = np.asarray(X)
        X = np.atleast_2d(X)  # Ensure X is 2D
        x0 = X[:, 0]  # shape (N,)
        x1 = X[:, 1]  # shape (N,)

        # 2D Gaussian around (mu1[0], mu1[1])
        g1 = np.exp(-(((x0 - self.mu1[0]) ** 2) / (2 * self.sigma1 ** 2)
                      + ((x1 - self.mu1[1]) ** 2) / (2 * self.sigma1 ** 2)))

        # 2D Gaussian around (mu2[0], mu2[1]) 
        g2 = np.exp(-(((x0 - self.mu2[0]) ** 2) / (2 * self.sigma2 ** 2)
                      + ((x1 - self.mu2[1]) ** 2) / (2 * self.sigma2 ** 2)))

        # Combine and normalize so max remains <= 1
        return (self.alpha1 * g1 + self.alpha2 * g2) / (self.alpha1 + self.alpha2)

# week 4 - extend to any number of dimensions and any number of peaks
class n_d_test:

    #def __init__(self, sigma1=0.1, sigma2=0.1, mu1=[0.3,0.3], mu2=[0.7,0.7], alpha1=1, alpha2=0.5):
    #    self.sigma1 = sigma1
    #    self.sigma2 = sigma2
    #    self.mu1 = mu1
    #    self.mu2 = mu2
    #    self.alpha1 = alpha1
    #    self.alpha2 = alpha2

    def __init__(self, sigma, mu, alpha):
        self.sigma = sigma
        self.mu = mu
        self.alpha = alpha

        assert len(sigma) == len(mu), "Sigma and mu array must have same length"
        assert len(sigma) == len(alpha), "Sigma and alpha array must have same length"
             

    # sigma: list of standard deviations
    # mu: list of mean points, each in n dimensions
    # alpha: list of peak heights

    sigma=[]
    mu=[]
    alpha=[]
    
    #sigma1 = 0.1
    #sigma2 = 0.1
    #mu1 = [0.3,0.3]
    #mu2 = [0.7,0.7]
    #alpha1 = 1
    #alpha2 = 0.5

    def call_function(self, X):
        """
        Bimodal function f(x0, x1) in [0, 1]^2 -> [0, 1].
        X is an array of shape (N, 2).
        Returns a 1D array of length N, each element in [0, 1].
        """

        X = np.asarray(X)
        if(len(X.shape)==1): # reshape 1d to 2d
            X = X.reshape(-1, 1)


        #X = np.atleast_2d(X)  # Ensure X is at least 2D
        gaussians = []
        dimensions = X.shape[1]
        #need 1 gaussian for each mu
        for m in range(len(self.mu)):
            terms = 0
            for d in range(dimensions):
                term = ((X[:, d] - self.mu[m][d]) ** 2) / (2 * self.sigma[m]**2)
                terms += term
            g = np.exp(-terms)

            gaussians.append(g)

        #normalise
        numerator = 0
        denominator = 0
        for m in range(len(self.mu)):
            numerator += (self.alpha[m] * gaussians[m])
            denominator += self.alpha[m]
        
        return numerator / denominator
