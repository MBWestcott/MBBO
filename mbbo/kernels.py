import numpy as np
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, Matern, DotProduct, WhiteKernel
import mbbo.functions

class KernelWithParams():
    def __init__(self, kernel, name, params):
        self.kernel = kernel
        self.name = name
        self.params = params


maternNu = 2.5

def make_kernels_for_tests(lengthscale):
    linearKernel = DotProduct(sigma_0=1.0)
    linearWithParams = KernelWithParams(linearKernel, "Linear (no noise)", {})
    rbfKernel = RBF(length_scale=lengthscale, length_scale_bounds=(0.001, 1))
    rbfWithParams = KernelWithParams(rbfKernel, "RBF", {})
    matern52Kernel = Matern(nu = maternNu, length_scale=lengthscale, length_scale_bounds=(0.001, 1))
    matern52WithParams = KernelWithParams(matern52Kernel, "Matern52", {"nu": maternNu})
    kernels = [linearWithParams, rbfWithParams, matern52WithParams]
    
    for alpha in np.arange(0.2, 2, 0.2):
        rqKernel = RationalQuadratic(alpha = alpha, length_scale=lengthscale, length_scale_bounds=(0.001, 1))
        kernels.append(KernelWithParams(rqKernel, "RationalQuadratic", {"alpha": alpha}))


    return kernels

def make_kernel_for_function(function_info:mbbo.functions.FunctionInfo):
    kernel = None
    match function_info.kernel_params["class"]:
        case "Linear (no noise)":
            kernel = DotProduct(sigma_0=1.0)
        case "RBF":
            kernel = RBF(length_scale=function_info.kernel_lengthscale, length_scale_bounds=(0.001, 1))
        case "Matern52":
            kernel = Matern(nu = function_info.kernel_params["nu"], length_scale=function_info.kernel_lengthscale, length_scale_bounds=(0.001, 1))
        case "RationalQuadratic":
            kernel = RationalQuadratic(alpha = function_info.kernel_params["alpha"], length_scale=function_info.kernel_lengthscale, length_scale_bounds=(0.001, 1))
        case _:
            raise Exception (f"Kernel class {function_info.kernel_params["class"]} not matched")

    return KernelWithParams(kernel, function_info.kernel_params["class"], function_info.kernel_params)