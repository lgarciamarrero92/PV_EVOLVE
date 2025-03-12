import numpy as np
from pymoo.core.problem import ElementwiseProblem
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator
class IVModel:
    def __init__(self,params={}):
        pass
    def calc(self,params,v_points,i_points):
        return self._calc(params,v_points,i_points)
    def _calc(self):
        raise NotImplementedError("Subclass must implement _calc method")


class IVFitting(ElementwiseProblem):

    def __init__(self,v_exp=None,i_exp=None,model=None,fitting_error="rmse",**kwargs):

        self.v_exp = v_exp
        self.i_exp = i_exp
        self.model = model
        self.fitting_error = fitting_error
        
        #Set bounds if exists
        for param in model.params:
            if model.bounds.get(param.name) is not None:
                param.bounds = model.bounds[param.name]

        xl = []
        xu = []
        vars = {}
        for param in model.params:
            vars[param.name] = param.var
            xl.append(param.bounds[0])
            xu.append(param.bounds[1])
        
        super().__init__(vars=vars,xl=np.array(xl),xu=np.array(xu), n_obj=1,**kwargs)
        
    def _evaluate(self, X, out, *args, **kwargs):
        eps = 1e-3
        if type(X) is dict:
            x = self.dict_to_list(X)
        else:
            x = X

        v_calc,i_calc = self.model.calc(x,self.v_exp,self.i_exp)
        valid_mask = ~np.isnan(v_calc) & ~np.isnan(i_calc)
        v_calc = v_calc[valid_mask]
        i_calc = i_calc[valid_mask]
        v_exp = self.v_exp[valid_mask]
        i_exp = self.i_exp[valid_mask]

        #Area between curves method
        if self.fitting_error == "area":
            if len(v_exp) < 2:
                out["F"] = np.inf
            else:
                out["F"] = relative_area_error(v_exp, i_exp, v_calc, i_calc)
        #RMSE Method
        else:
            if len(v_exp) < 2:
                out["F"] = np.inf
            else:
                out["F"] = np.sqrt((np.square(i_exp - i_calc)).mean()) + np.sqrt((np.square(v_exp - v_calc)).mean())

    def residuals(self, X):
        v_calc,i_calc = self.model.calc(X,self.v_exp,self.i_exp)
        return (self.i_exp - i_calc)
    
    def comparisonPlot(self, X, title=None,interpolate=False):
        
        if type(X) is dict:
            x = self.dict_to_list(X)
        else:
            x = X

        if title is None:
            title = "Experimental vs Fitting (" + self.model.name + ")"

        fig = plt.figure()

        v_exp = self.v_exp
        i_exp = self.i_exp
        num_points = 1000

        v_points = np.linspace(np.min(v_exp),np.max(v_exp),num_points)
        i_points = np.linspace(np.min(i_exp),np.max(i_exp),num_points)

        v_calc,i_calc = self.model.calc(x,v_points,i_points)

        valid_mask = ~np.isnan(v_calc) & ~np.isnan(i_calc)
        v_calc = v_calc[valid_mask]
        i_calc = i_calc[valid_mask]

        plt.title(title)
        plt.xlabel("$V \ [V]$")
        plt.ylabel("$I \ [A]$")
        plt.scatter(v_exp,i_exp,label="Experimental")
        if len(v_calc) > 0:
            mpp_ind = np.argmax(v_calc*i_calc)
            plt.plot(v_calc,i_calc,label="Fitting", color='r')
            plt.scatter(v_calc[mpp_ind],i_calc[mpp_ind],label="MPP", color='black')
        plt.legend()
        return fig
    
    def formatSol(self, X, latex=False):
        if type(X) is dict:
            x = self.dict_to_list(X)
        else:
            x = X
        
        out = dict()
        self._evaluate(X, out)

        values = np.append(x,out['F'])
        
        #Custom formating in Model
        if hasattr(self.model, 'formatSol'):
            df = self.model.formatSol(x,out['F'])
        else:
            cols = []
            for param in self.model.params:
                if latex == True and param.latex_name is not None:
                    cols.append(param.latex_name)
                else:
                    cols.append(param.name)

            cols.append('Error')

            df = pd.DataFrame(np.atleast_2d(values),columns=cols,index=["Values"])

        return df
    
    def dict_to_list(self,X):
        x = []
        for param in self.model.params:
            x.append(X[param.name])
        return x

def relative_area_error(v_exp, i_exp, v_calc, i_calc):
    """
    Calculate the relative area error between experimental and calculated data.
    
    This function handles cases where either voltage or current is the same
    between experimental and calculated data and where the data may not be sorted.

    Parameters:
    v_exp  : numpy array of experimental voltage values.
    i_exp  : numpy array of experimental current values.
    v_calc : numpy array of calculated voltage values.
    i_calc : numpy array of calculated current values.

    Returns:
    relative_area_error : float
    """
    # Case 1: Voltage is the same between experimental and calculated data
    if np.array_equal(v_exp, v_calc):
        # Sort by voltage
        sorted_indices = np.argsort(v_exp)
        v_exp_sorted = v_exp[sorted_indices]
        i_exp_sorted = i_exp[sorted_indices]
        i_calc_sorted = i_calc[sorted_indices]
        
        # Calculate the absolute difference between the currents
        current_diff = np.abs(i_exp_sorted - i_calc_sorted)
        
        # Integrate this difference with respect to voltage
        area_diff = np.trapz(current_diff, v_exp_sorted)
        area_exp = np.trapz(i_exp_sorted, v_exp_sorted)
    
    # Case 2: Current is the same between experimental and calculated data
    elif np.array_equal(i_exp, i_calc):
        # Sort by current
        sorted_indices = np.argsort(i_exp)
        i_exp_sorted = i_exp[sorted_indices]
        v_exp_sorted = v_exp[sorted_indices]
        v_calc_sorted = v_calc[sorted_indices]
        
        # Calculate the absolute difference between the voltages
        voltage_diff = np.abs(v_exp_sorted - v_calc_sorted)
        
        # Integrate this difference with respect to current
        area_diff = np.trapz(voltage_diff, i_exp_sorted)
        area_exp = np.trapz(v_exp_sorted, i_exp_sorted)
    
    else:
        raise ValueError("Neither voltage nor current are consistent between experimental and calculated data.")
    
    # Calculate relative area error
    relative_area_error = area_diff / np.abs(area_exp)
    
    return relative_area_error