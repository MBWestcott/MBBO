
import numpy as np

week1_out = [0.0, -0.03634716524130564, -0.13995571712281177, -11.512791229057324, 351.7115420928652, -0.5971511450896173, 0.2910786825809617, 8.618272750952901]
week1_in = [np.array([0.00367, 0.9999 ]), 
            np.array([0.851999, 0.973204]), 
            np.array([0.747032, 0.28413 , 0.226329]), 
            np.array([0.169128, 0.756136, 0.275457, 0.528761]), 
            np.array([0.439601, 0.772709, 0.376277, 0.933269]), 
            np.array([0.232204, 0.132714, 0.53824 , 0.760706, 0.075595]), 
            np.array([0.476821, 0.248196, 0.242816, 0.576157, 0.162416, 0.290926]), 
            np.array([0.221603, 0.703755, 0.674607, 0.130295, 0.376739, 0.669444, 0.136655, 0.061316])]    

week2_out = [-1.2075460499722905e-18, 0.17608630702211278, -0.17239781799687137, -31.982880235497266, 1236.8846557000643, -2.451406055102475, 0.00010805707939840242, 5.178959940699899]
week2_in = [np.array([0.476035, 0.572563]), 
            np.array([0.641846, 0.498841]), 
            np.array([0., 0., 0.]), 
            np.array([0.953433, 0.895217, 0.812477, 0.618719]), 
            np.array([0.987523, 0.470227, 0.946409, 0.105412]), 
            np.array([3.40696e-01, 4.94179e-01, 2.10000e-05, 3.08050e-02, 9.39958e-01]), 
            np.array([0.88314 , 0.756642, 0.      , 0.      , 0.9     , 0.942719]), 
            np.array([0.993634, 0.968223, 0.979285, 0.397318, 0.965856, 0.955218, 0.006078, 0.024001])]

week3_out = [-2.118633970077695e-95, -0.1068943462941895, -0.005838531351604155, -2.6718044713157307, 32.0025, -1.4580645404618957, 0.4892165178828796, 9.9417237968706]

week3_in = [np.array([0.127849, 0.198491]), 
            np.array([0.246077, 0.656597]), 
            np.array([0.492581, 0.611593, 0.5     ]), 
            np.array([0.510358, 0.521985, 0.383995, 0.445439]), 
            np.array([0.5, 0.5, 0.5, 0.5]), 
            np.array([0.66336 , 0.      , 0.999999, 0.332984, 0.      ]), 
            np.array([0.      , 0.165185, 0.28681 , 0.      , 0.318109, 0.999999]), 
            np.array([0.119265, 0.254466, 0.117275, 0.24563 , 0.548426, 0.553172,  0.230111, 0.516062])]

week4_out = [-8.306597721001677e-27, 0.715790799340666, -0.00506242600241439, -3.2126105576284227, 31.94090504001378, -0.9205277885179105, 0.3911680928412005, 9.6899612812574]

week4_in = [np.array([0.24001 , 0.357107]),
                np.array([0.5, 0.5]), 
                np.array([0.5, 0.5, 0.5]), 
                np.array([0.549669, 0.508442, 0.413776, 0.413008]), 
                np.array([0.500102, 0.500102, 0.500102, 0.500102]), 
                np.array([0.563405, 0.      , 0.83134 , 0.999999, 0.      ]), 
                np.array([0.      , 0.626234, 0.280125, 0.      , 0.36777 , 0.451863]), 
                np.array([0.275027, 0.304704, 0.160147, 0.328388, 0.419169, 0.578759, 0.436166, 0.614079])]

week5_out = [1.517648729565899e-192, 0.509599138595138, -0.025681820315624142, -4.078914281244423, 629.9338529410855, -1.747233852094004, 0.39256467139392903, 9.7675674964181]

week5_in = [np.array([0.999999, 0.999999]), np.array([0.666698, 0.666698]), np.array([0.558875, 0.558874, 0.558875]), np.array([0.523385, 0.494608, 0.22783 , 0.357468]), np.array([0.932544, 0.415248, 0.89143 , 0.050433]), np.array([0.      , 0.687353, 0.      , 0.999999, 0.      ]), np.array([0.      , 0.56895 , 0.354465, 0.290165, 0.482077, 0.989692]), np.array([0.      , 0.047944, 0.315163, 0.115808, 0.571106, 0.59513 ,
       0.376754, 0.548807])]

week6_out = [-1.7808346077779874e-113, 0.8097867781489138, -0.021754112476429704, -2.3290209829198107, 1238.0344144400913, -0.3531644651233685, 1.4207771820533106, 9.863194426458]

week6_in = [np.array([0.999999, 0.784908]), 
            np.array([0.500257, 0.500039]), 
            np.array([0.500001, 0.500004, 0.499999]), 
            np.array([0.502787, 0.48848 , 0.355693, 0.387261]), 
            np.array([0.987554, 0.470163, 0.946567, 0.105327]), 
            np.array([0.463126, 0.317874, 0.508172, 0.723817, 0.144808]), np.array([0.055316, 0.488299, 0.249433, 0.216093, 0.410181, 0.731049]), np.array([0.108893, 0.287056, 0.194225, 0.299992, 0.537696, 0.356217,
       0.306504, 0.37132 ])]

responses = [week1_out, week2_out, week3_out, week4_out, week5_out, week6_out]


def get_function_data(function_number):
    ary_in = np.load(f'../data/raw/initial_data/function_{function_number}/initial_inputs.npy')
    ary_out = np.load(f'../data/raw/initial_data/function_{function_number}/initial_outputs.npy')

    
    ary_out=np.append(ary_out, week1_out[function_number-1])
    ary_out=np.append(ary_out, week2_out[function_number-1])
    ary_out=np.append(ary_out, week3_out[function_number-1])
    ary_out=np.append(ary_out, week4_out[function_number-1])
    ary_out=np.append(ary_out, week5_out[function_number-1])
    ary_out=np.append(ary_out, week6_out[function_number-1])
    ary_in=np.vstack((ary_in, week1_in[function_number-1]))
    ary_in=np.vstack((ary_in, week2_in[function_number-1]))
    ary_in=np.vstack((ary_in, week3_in[function_number-1]))
    ary_in=np.vstack((ary_in, week4_in[function_number-1]))
    ary_in=np.vstack((ary_in, week5_in[function_number-1]))
    ary_in=np.vstack((ary_in, week6_in[function_number-1]))
    
    return ary_in, ary_out

class FunctionInfo():
    # week 7 - function 1 - need to find 2nd optimum which seems to be around [0.9, 0.2] - keep exploring around there (changed kappa from 0.1 to 0.8)
    # function 2 - "lot of local optima" so keep exploring too, especially unexplored areas - top left (around [0.3, 0.85]) and bottom right (around [0.775, 0.275])
    rbf_lengthscales = [0.8, 0.5, 0.016, 0.337, 0.0162, 0.68, 0.644, 1.0 ] # functions 2 and 8 didn't find optimal lengthscale in training
    ucb_kappas = [0.8, 0.5, 0.00001, 0.8, 0.4, 0.8, 0.8, 0.8] # only got successful calibration for first 3. Default to 0.8 for the rest (high because still exploring) 
                                                              # - except function 5 which is unimodal so can exploit more.
                                                              # Function 3 - aim is to reduce bad side effects of drug combination - have a maximum around 0.5,0.5,0.5 so from week 5 on, exploiting that
    perturb_max_starts = [0,0,0,0,-0.055,0,0,0] #having trouble getting function 5 to explore a little more away from its maximum - nudge
    kernel_params_list=[{"class": "RationalQuadratic", "alpha": 0.4},
                   {"class": "RationalQuadratic", "alpha": 1.0},
                   {"class": "RationalQuadratic", "alpha": 1.0},
                   {"class": "RationalQuadratic", "alpha": 1.0}, #f4 - all NaN
                   {"class": "RationalQuadratic", "alpha": 1.4}, #f5 = mainly NaN
                   {"class": "Matern52", "nu":2.5}, #f6 - mainly NaN
                   {"class": "RationalQuadratic", "alpha": 1.0},
                   {"class": "RationalQuadratic", "alpha": 1.0}]

    def __init__(self, function_number):
        self.function_number = function_number
        function_ix = function_number - 1
        self.kernel_lengthscale = self.rbf_lengthscales[function_ix]
        self.ucb_kappa = self.ucb_kappas[function_ix]
        self.kernel_params = self.kernel_params_list[function_ix]
        self.perturb_max_start = self.perturb_max_starts[function_ix]