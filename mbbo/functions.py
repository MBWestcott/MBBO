
import numpy as np

queries = [
    #week1_in, week2_in, week3_in, week4_in, week5_in, week6_in, week7_in, week8_in, week9_in, week10_in, week11_in, week12_in, week13_in, week14_in, week15_in, week16_in
    ]
responses = [
    #week1_out, week2_out, week3_out, week4_out, week5_out, week6_out, week7_out, week8_out, week9_out, week10_out, week11_out, week12_out, week13_out, week14_out, week15_out, week16_out
    ]

def load_initial_data(function_number: int, include_set2: bool = True):
    ary_in = np.load(f'../data/raw/initial_data/function_{function_number}/initial_inputs.npy')
    ary_out = np.load(f'../data/raw/initial_data/function_{function_number}/initial_outputs.npy')
    if(include_set2):
        ary2_in = np.load(f'../data/raw/initial_data2/function_{function_number}/initial_inputs.npy')
        ary2_out = np.load(f'../data/raw/initial_data2/function_{function_number}/initial_outputs.npy')
        ary_in = np.vstack((ary_in, ary2_in))
        ary_out = np.append(ary_out, ary2_out)

    return ary_in, ary_out

def get_function_data(function_number: int, include_set2: bool = True, include_observed: bool = True):
    ary_in, ary_out = load_initial_data(function_number, include_set2)
    if include_observed:    
        for r in responses:
            ary_out = np.append(ary_out, r[function_number - 1])
        for q in queries:
            ary_in = np.vstack((ary_in, q[function_number - 1]))
    return ary_in, ary_out

class FunctionInfo():
    # week 7 - function 1 - need to find 2nd optimum which seems to be around [0.9, 0.2] - keep exploring around there (changed kappa from 0.1 to 0.8)
    # function 2 - "lot of local optima" so keep exploring too, especially unexplored areas - top left (around [0.3, 0.85]) and bottom right (around [0.775, 0.275])
    rbf_lengthscales = [0.00421, 0.0544, 0.016, 3.51, 0.0942, 3.04, 0.198, 2.38 ] # functions 2 and 8 didn't find optimal lengthscale in training
    default_lengthscale_lb = 0.001
    default_lengthscale_ub = 20.0
    lengthscale_bounds_list = [(default_lengthscale_lb, default_lengthscale_ub)] * 8
    lengthscale_bounds_list[0] = (0.0001, 20) #f1
    lengthscale_bounds_list[1] = (0.0001, 50)
    lengthscale_bounds_list[2] = (0.0001, 80) #f3
    constant_value_bounds_list = [(1e-5, 1e5)] * 8 # constant kernel default value bounds
    constant_value_bounds_list[7] = (1e-5, 1e7) #f8
    ucb_kappas = [0.8, 0.5, 0.2, 0.7, 0.1, 0.7, 0.4, 0.4] # lowered for exploitation excepr 1,2,4,5
    #ucb_kappas = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    #ucb_kappas = ucb_kappas * 4 # increase for random forest regressor
    #ucb_kappas = [2, 0.5, 0.00001, 0.8, 0.4, 0.8, 0.8, 0.8]
    #ucb_kappas = [0.8, 0.5, 0.00001, 0.8, 0.4, 0.8, 0.8, 0.8] # only got successful calibration for first 3. Default to 0.8 for the rest (high because still exploring) 
                                                              # - except function 5 which is unimodal so can exploit more.
                                                              # Function 3 - aim is to reduce bad side effects of drug combination - have a maximum around 0.5,0.5,0.5 so from week 5 on, exploiting that
    acq_xis = [2.9, 2.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9] # default to 0.1 for all functions                                                    
    #perturb_max_starts = [0,0,0,0,0.055,0,0,0.1] #having trouble getting function 5 to explore a little more away from its maximum - nudge
    perturb_max_starts = [0,0,0,0,0,0,0,0] 
    kernel_params_list=[{"class": "RationalQuadratic", "alpha": 0.949, "alphabounds": (1e-5, 1e5)},
                   {"class": "RationalQuadratic", "alpha": 2.0, "alphabounds": (1e-8, 1e5)},
                   {"class": "Linear (no noise)", "alpha": 1.0, "alphabounds": (1e-5, 1e5)},
                   {"class": "RationalQuadratic", "alpha": 0.6, "alphabounds": (1e-5, 1e5)},
                   {"class": "RationalQuadratic", "alpha": 0.2, "alphabounds": (1e-5, 1e5)}, #f5 = mainly NaN
                   {"class": "RationalQuadratic", "alpha":0.0417, "alphabounds": (1e-5, 1e5), "nu":2.5}, #f6 - mainly NaN
                   {"class": "RationalQuadratic", "alpha": 0.308, "alphabounds": (1e-5, 1e5)},
                   {"class": "RBF", "alpha": 1.0, "alphabounds": (1e-5, 1e7)}]

    def __init__(self, function_number):
        self.function_number = function_number
        function_ix = function_number - 1
        self.kernel_lengthscale = self.rbf_lengthscales[function_ix]
        self.ucb_kappa = self.ucb_kappas[function_ix]
        self.acq_xi = self.acq_xis[function_ix]
        self.kernel_params = self.kernel_params_list[function_ix]
        self.perturb_max_start = self.perturb_max_starts[function_ix]
        self.n_restarts_optimizer = 0 if self.kernel_params["class"] in ["RBF", "Linear (no noise)"] else 10 # use restarts optimizer for more complex kernels
        self.lengthscale_bounds = self.lengthscale_bounds_list[function_ix]
        self.constant_value_bounds = self.constant_value_bounds_list[function_ix]


def scale(function_num, values):
    #function 1 - values mostly very close to 0
    if function_num != 1:
        return values
    
    # Step 1: Replace zeros with a small positive value to avoid log(0)
    epsilon = 1e-250  # A very small number
    values_safe = np.where(values == 0, epsilon, values)

    # Step 2: Take the logarithm of absolute values
    log_values = np.log10(np.abs(values_safe))

    # Step 3: Normalize to the range [-1, 1]
    log_min, log_max = np.min(log_values), np.max(log_values)
    scaled_values = 2 * (log_values - log_min) / (log_max - log_min) - 1

    # Step 4: Restore signs
    return np.sign(values) * np.abs(scaled_values)

def reduce(function_num, values):
    """
    Reduce the number of dimensions to the most important ones for the given function
    """
    important_dimensions = {}
    important_dimensions = {8:[0,2,6]} # function 8: based on coefficients [-1.6245557  -0.46852827 -2.36121139 -0.50061563  0.15028904 -0.18675851 -1.20356151  0.24547701]
    if function_num not in important_dimensions:
        return values
    return values[:, important_dimensions[function_num]]

def is_close(new_x, existing_x, threshold = 0.1):
    distances = np.linalg.norm(existing_x - new_x, axis=1)
    min_distance = distances.min()
    closest_existing = existing_x[distances.argmin()]
    
    too_close = min_distance < threshold
    
    return (too_close, min_distance, closest_existing)
    
    