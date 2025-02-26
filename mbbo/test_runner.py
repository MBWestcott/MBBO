import numpy as np
import random
from mbbo.test_gaussian import one_d_test, two_d_test, n_d_test
import mbbo.acquisition
import mbbo.functions
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy import optimize

noise_assumption = 1e-10 # noise assumption, a hyper-parameter

# Test acquisition functions on the 1d function

input_bounds = [(0, 1)] # bounds for the input space

def bounds_midpoint(input_bounds):
    return np.array([(low + high) / 2.0 for low, high in input_bounds])

def test_on_oned(rbf_lengthscale, ucb_acquisition_kappa, test_oned_function, max_iterations):
    
    objective_x = np.linspace(0, 1, 500)
    objective_y = test_oned_function.call_function(objective_x)
    objective_y_max = max(objective_y)

    kernel = RBF(length_scale=rbf_lengthscale, length_scale_bounds='fixed')
    model = GaussianProcessRegressor(kernel = kernel, alpha=noise_assumption)

    X = []
    Y = []
    # First point
    x0 = bounds_midpoint(input_bounds)  # array([0.5])
    X.append(x0)
    Y.append(test_oned_function.call_function(x0))
    i=0
    #print("Objective max:", objective_y_max)
    while abs(max(Y) - objective_y_max) > 0.01:
        i+=1
        if i > max_iterations:
            print("Max iterations reached")
            return 0
        #print("Iteration", i, " Max found:", max(Y))
        # print(abs(max(Y) - objective_y_max))
        # fit the model
        model.fit(X, Y)
            
        # optimize the acquisition function
        result = optimize.minimize(lambda x: -mbbo.acquisition.UCB(x, model, ucb_acquisition_kappa), x0=bounds_midpoint(input_bounds), bounds=input_bounds)
        x_new = result.x
        y_new = test_oned_function.call_function(x_new)
        
        # add the new observation to the training set
        X.append(x_new) #assumes 1d
        Y.append(y_new)

    return i

#test on 2d functions
def test_on_twod(rbf_lengthscale, ucb_acquisition_kappa, test_twod_function, max_iterations):
    
    twod_input_bounds = [(0, 1), (0,1)]

    N = 100
    x0_vals = np.linspace(0, 1, N)
    x1_vals = np.linspace(0, 1, N)
    X0, X1 = np.meshgrid(x0_vals, x1_vals)        # shape (N, N) each
    objective_x = np.column_stack((X0.ravel(), X1.ravel()))  # shape (N*N, 2)
    objective_y = test_twod_function.call_function(objective_x)
    objective_y_max = max(objective_y)

    kernel = RBF(length_scale=rbf_lengthscale)
    model = GaussianProcessRegressor(kernel = kernel, alpha=noise_assumption)

    X = []
    Y = []
    # First point
    initial_x = bounds_midpoint(twod_input_bounds)  # array([0.5])
    X.append(initial_x)
    Y.append(test_twod_function.call_function(initial_x))
    for i in range(5):
        starting_x = [random.random(), random.random()]
        starting_y = test_twod_function.call_function(starting_x)
        X.append(starting_x)
        Y.append(starting_y)
    
    i=0

    #print("Objective max:", objective_y_max)
    while abs(max(Y) - objective_y_max) > 0.01:
        i+=1
        # print("Max y:", max(Y))
        if i > max_iterations:
            print("Max iterations reached")
            return 0
        #print("Iteration", i, " Max found:", max(Y))
        # print(abs(max(Y) - objective_y_max))
        # fit the model
        model.fit(X, Y)
            
        # optimize the acquisition function
        result = optimize.minimize(lambda x: -mbbo.acquisition.UCB(x, model, ucb_acquisition_kappa), x0=initial_x, bounds=twod_input_bounds)
        x_new = result.x
        y_new = test_twod_function.call_function(x_new)
        
        # add the new observation to the training set
        X.append(x_new)
        Y.append(y_new)

    return i

# week 4 - set up test profiles to test the higher dimensions

#Generate a random data point between 0 and 1 for the number of dimensions
def random_point(dimensions):
    return np.random.rand(dimensions)

class TestProfile():
    def __init__(self, dimensions, start_samples, maxima, std = 0.1, no_calibration_functions = 4):
        self.dimensions = dimensions
        self.start_samples = start_samples # number of samples to start with
        self.maxima = maxima
        self.std = std
        self.no_calibration_functions = no_calibration_functions

    def CreateFunctions(self):
        profile_functions = []
        for i in range(self.no_calibration_functions):
            mu = []
            # assume 0.1 std for all gaussians
            # assume alpha (relative height) = 1 for first peak and 0.5 for others
            sigma = [self.std] * self.maxima
            alpha = [1]

            for m in range(self.maxima):
                mu.append(random_point(self.dimensions))
                if(m > 0):
                    alpha.append(0.5)
            ndt = n_d_test(mu=mu, sigma=sigma, alpha=alpha)
            profile_functions.append (ndt)
        return profile_functions

def get_test_profile(function_number):
    X, Y = mbbo.functions.get_function_data(function_number)
    # Return a test profile matching what we know about the objective function
    # Function 1 (2d): 2 maxima
    # Function 2 (2d): "a lot of local optima" - use small standard deviation (0.05) and 10 peaks
    # Function 4 (4d): "a lot of local optima" - use small standard deviation (0.05) and 10 peaks (reduced from 20 in week 7 as not getting anywhere!)
    # Function 5 (4d): 1 maximum
    # For other functions - use the number of dimensions as the number of peaks
    dimensions = X.shape[1]
    start_samples = X.shape[0]
    maxima = dimensions
    std = 0.15
    if(function_number == 1):
        maxima = 2
    elif(function_number ==2):
        std = 0.1
        maxima=10
    elif(function_number == 4):
        std = 0.05
        maxima=20
    elif(function_number==5):
        maxima=1

    no_calibration_functions = 4 #number of test Gaussians to use for calibration. Use 4 for first 4, 3 for next 2, then just 1 for last two because they are very slow to run.
    if function_number >= 7:
        no_calibration_functions = 1
    elif function_number <= 5:
        no_calibration_functions = 3

    return TestProfile(dimensions, start_samples, maxima, std, no_calibration_functions)

def input_bounds_for_dim(dimensions):
    return [(0, 1) for _ in range(dimensions)]

def test_on_n_d(test_profile: TestProfile, kernel, ucb_acquisition_kappa, test_function: n_d_test, max_iterations, n_grid = 20):
    
    # 1. Create a list of coordinate arrays, one for each dimension.
    coords_1d = [np.linspace(0, 1, n_grid) for _ in range(test_profile.dimensions)]

    # 2. Create the D-dimensional mesh.
    #    Each element of `mesh` will be an array of shape (N, N, ..., N) [D times].
    mesh = np.meshgrid(*coords_1d, indexing='ij')

    # 3. Flatten each dimension, then stack them to get shape (N^D, D).
    #    .ravel() flattens the array, and column_stack collects them into columns.
    objective_x = np.column_stack([m.ravel() for m in mesh])

    objective_y = test_function.call_function(objective_x)
    objective_y_max = max(objective_y)

    model = GaussianProcessRegressor(kernel = kernel, alpha=noise_assumption)

    X = []
    Y = []
    # First point
    bounds = input_bounds_for_dim(test_profile.dimensions)
    initial_x = bounds_midpoint(bounds)  
    #X.append(initial_x)
    #Y.append(test_function.call_function(initial_x))
    for i in range(test_profile.start_samples):
        starting_x = random_point(test_profile.dimensions)
        starting_y = test_function.call_function([starting_x])
        X.append(starting_x)
        Y.append(starting_y[0])
    
    i=0
    #print("Objective max:", objective_y_max)
    while abs(max(Y) - objective_y_max) > 0.01:
        i+=1
        # print("Max y:", max(Y))
        if i > max_iterations:
            #print("Max iterations reached")
            return np.nan, np.nan
        #print("Iteration", i, " Max found:", max(Y))
        # print(abs(max(Y) - objective_y_max))
        # fit the model
        model.fit(X, Y)
            
        # optimize the acquisition function
        result = optimize.minimize(lambda x: -mbbo.acquisition.UCB(x, model, ucb_acquisition_kappa), x0=initial_x, bounds=bounds)
        x_new = result.x
        y_new = test_function.call_function([x_new])
        
        # add the new observation to the training set
        X.append(x_new)
        Y.append(y_new[0])
    trained_length_scale = 0
    if hasattr(model, "kernel_"):
        if(hasattr(model.kernel_, "length_scale")):
            trained_length_scale = model.kernel_.length_scale
    # print (trained_length_scale)
    return i, trained_length_scale
