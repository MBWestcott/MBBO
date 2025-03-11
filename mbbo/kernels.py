import numpy as np
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, Matern, DotProduct, WhiteKernel, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
import mbbo.functions
from mbbo.functions import FunctionInfo, get_function_data

class KernelWithParams():
    def __init__(self, kernel, name, params):
        self.kernel = kernel
        self.name = name
        self.params = params


maternNu = 2.5

def make_kernels_for_tests(lengthscale, lengthscale_lb = 0.001, lengthscale_ub = 20.0):
    lsb = (lengthscale_lb, lengthscale_ub)
    linearKernel = DotProduct(sigma_0=1.0)
    linearWithParams = KernelWithParams(linearKernel, "Linear (no noise)", {})
    rbfKernel = RBF(length_scale=lengthscale, length_scale_bounds=lsb)
    rbfWithParams = KernelWithParams(rbfKernel, "RBF", {})
    matern52Kernel = Matern(nu = maternNu, length_scale=lengthscale, length_scale_bounds=lsb)
    matern52WithParams = KernelWithParams(matern52Kernel, "Matern52", {"nu": maternNu})
    kernels = [linearWithParams, rbfWithParams, matern52WithParams]
    
    for alpha in np.arange(0.2, 2, 0.2):
        rqKernel = RationalQuadratic(alpha = alpha, length_scale=lengthscale, length_scale_bounds=(0.001, 1))
        kernels.append(KernelWithParams(rqKernel, "RationalQuadratic", {"alpha": alpha}))


    return kernels

def make_kernels_with_const(sigma = 1.0, lengthscale_lb = 0.001, lengthscale_ub = 20.0):
    lsb = (lengthscale_lb, lengthscale_ub)
    #return {
    #    "RBF": RBF(length_scale=1.0, length_scale_bounds=lsb),
    #    "Matern_1.5": Matern(length_scale=1.0, nu=1.5, length_scale_bounds=lsb),
    #    "Matern_2.5": Matern(length_scale=1.0, nu=2.5, length_scale_bounds=lsb),
    #    "RationalQuadratic": RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=lsb),
    #    "DotProduct": DotProduct(sigma_0=sigma)}


    return {
    #"RBF": C(1.0) * RBF(length_scale=1.0, length_scale_bounds=lsb),
    "Matern_1.5": C(1.0) * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=lsb),
    "Matern_2.5": C(1.0) * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=lsb),
    "RationalQuadratic": C(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=lsb),
    #"DotProduct": DotProduct(sigma_0=sigma) + C(1.0)
    }

def make_kernel_for_function(function_info:mbbo.functions.FunctionInfo, sigma = 1.0):
    kernel = None
    x = 2
    if x==1:
        match function_info.kernel_params["class"]:
            case "Linear (no noise)":
                kernel = DotProduct(sigma_0=sigma)
            case "RBF":
                kernel = RBF(length_scale=function_info.kernel_lengthscale, length_scale_bounds=function_info.lengthscale_bounds)
            case "Matern52":
                kernel = Matern(nu = function_info.kernel_params["nu"], length_scale=function_info.kernel_lengthscale, length_scale_bounds=function_info.lengthscale_bounds)
            case "RationalQuadratic":
                kernel = RationalQuadratic(alpha = function_info.kernel_params["alpha"], length_scale=function_info.kernel_lengthscale, length_scale_bounds=function_info.lengthscale_bounds)
            case _:
                raise Exception (f"Kernel class {function_info.kernel_params["class"]} not matched")
    else: # with constant kernel
        match function_info.kernel_params["class"]:
            case "Linear (no noise)":
                kernel = DotProduct(sigma_0=sigma) + C(1.0)
            case "RBF":
                # can't get the constant kernel to work with RBF
                #kernel = RBF(length_scale=function_info.kernel_lengthscale, length_scale_bounds=function_info.lengthscale_bounds)
                kernel = C(1.0) * RBF(length_scale=function_info.kernel_lengthscale, length_scale_bounds=function_info.lengthscale_bounds)
            case "Matern52":
                kernel = C(1.0) * Matern(nu = function_info.kernel_params["nu"], length_scale=function_info.kernel_lengthscale, length_scale_bounds=function_info.lengthscale_bounds)
            case "RationalQuadratic":
                kernel = C(1.0) * RationalQuadratic(alpha = function_info.kernel_params["alpha"], length_scale=function_info.kernel_lengthscale, length_scale_bounds=function_info.lengthscale_bounds)
            case _:
                raise Exception (f"Kernel class {function_info.kernel_params["class"]} not matched")

    return KernelWithParams(kernel, function_info.kernel_params["class"], function_info.kernel_params)


def choose_kernel(function_number):
    print(f"Function {function_number}")
    info_f = FunctionInfo(function_number)
    X,y = get_function_data(function_number)
    if(function_number == 1):
        y = mbbo.functions.scale_f1(y)
    ks = make_kernels_with_const(lengthscale_lb=info_f.lengthscale_bounds[0], lengthscale_ub=info_f.lengthscale_bounds[1])
    results = {}
    for name, kernel in ks.items():
        # Dictionary to store results
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.05, n_restarts_optimizer=10)

        gp.fit(X, y)
        
        # Compute Log Marginal Likelihood (MLL)
        mll = gp.log_marginal_likelihood_value_
        results[name] = {"MLL": mll, "Kernel": gp.kernel_}

        print(f"🔹 Kernel: {name}, MLL: {mll:.3f}, Optimized Kernel: {gp.kernel_}")

    # Select the best kernel based on MLL closest to 0
    best_kernel_name = min(results, key=lambda k: abs(results[k]["MLL"]))
    best_kernel = results[best_kernel_name]["Kernel"]

    print(f"\n✅ Best Kernel: {best_kernel_name} with MLL: {results[best_kernel_name]['MLL']:.3f}")
    return best_kernel, results[best_kernel_name]['MLL']

def choose_kernel_3():
#Function 3 - "one dimension may have no effect" - investigate kernels that omit one dimension"""
    info_f = FunctionInfo(3)
    X,y = get_function_data(3)
    for i in range(3):
        print(f"Omitting dimension {i}")
        X_ = np.delete(X, i, 1)
        ks = make_kernels_with_const(lengthscale_lb=info_f.lengthscale_bounds[0], lengthscale_ub=info_f.lengthscale_bounds[1])
        results = {}
        for name, kernel in ks.items():
            # Dictionary to store results
            gp = GaussianProcessRegressor(kernel=kernel, alpha=0.05, n_restarts_optimizer = info_f.n_restarts_optimizer)

            gp.fit(X_, y)
            
            # Compute Log Marginal Likelihood (MLL)
            mll = gp.log_marginal_likelihood_value_
            results[name] = {"MLL": mll, "Kernel": gp.kernel_}

            print(f"🔹 Kernel: {name}, MLL: {mll:.3f}, Optimized Kernel: {gp.kernel_}")

        # Select the best kernel based on MLL closest to 0
        best_kernel_name = min(results, key=lambda k: abs(results[k]["MLL"]))
        best_kernel = results[best_kernel_name]["Kernel"]
        print(f"\n✅ Best Kernel: {best_kernel_name} with MLL: {results[best_kernel_name]['MLL']:.3f}")
        print("") 





