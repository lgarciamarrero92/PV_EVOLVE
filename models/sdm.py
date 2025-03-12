import numpy as np
from scipy.special import lambertw
from core.params import Iph, Isat, eta, Rs, Rsh, nNsVth
from core.iv import IVModel
from utils.constants import BOLTZMANN_CONSTANT, ELECTRON_VOLT
from utils.helpers import celsius_to_kelvin
from scipy.optimize import newton
from scipy.optimize import brentq
import pandas as pd

class SDM(IVModel):
    def __init__(self,tm=None,ns=None,method="lambertw_i_from_v",bounds={}):
        self.name = "SDM"
        self.method = method
        self.bounds = bounds
        self.Vt = None
        self.tm = tm
        
        if tm is not None and ns is not None:
            #Calculus of Vt (Thermal voltage)
            T=celsius_to_kelvin(tm)
            self.Vt=(ns*BOLTZMANN_CONSTANT*T)/ELECTRON_VOLT

            self.params = [
                Iph(),
                Isat(),
                eta(),
                Rs(),
                Rsh()
            ]
        else:
            self.params = [
                Iph(),
                Isat(),
                nNsVth(),
                Rs(),
                Rsh()
            ]

        super().__init__(params=self.params)

    def _calc(self,params,v_points=None,i_points=None):
        
        if self.Vt is not None:
            Iph, Isat, eta, Rs, Rsh = params
            nNsVth = self.Vt*eta
        else:
            Iph, Isat, nNsVth, Rs, Rsh = params

        if self.method == 'lambertw_v_from_i':
            v_calc = self.lambertw_v_from_i(Iph, Isat, nNsVth, Rs, Rsh, i_points)
            return (v_calc,i_points)
        
        elif self.method == 'lambertw_i_from_v':
            i_calc = self.lambertw_i_from_v(Iph, Isat, nNsVth, Rs, Rsh, v_points)
            return (v_points,i_calc)
        
        elif self.method == 'implicit_v_from_i':
            v_calc = self.implicit_v_from_i(Iph, Isat, nNsVth, Rs, Rsh, i_points)
            return (v_calc,i_points)
        
        elif self.method == 'implicit_i_from_v':
            i_calc = self.implicit_i_from_v(Iph, Isat, nNsVth, Rs, Rsh, v_points)
            return (v_points,i_calc)
        else:
            raise NotImplementedError("Method not implemented")
    
    def lambertw_i_from_v(self, Iph, Isat, nNsVth, Rs, Rsh, v_points):

        Teta1=Rsh*Rs/(Rsh+Rs)/(nNsVth)*Isat
        Teta2=(Rsh*Rs*(Iph+Isat)+Rsh*v_points)/(nNsVth*(Rsh+Rs))

        # overflow is explicitly handled below, so ignore warnings here
        with np.errstate(over='ignore'):
            Teta=Teta1*np.exp(Teta2)
        
        # may overflow to np.inf
        lambertwterm = lambertw(Teta).real

        # Record indices where lambertw input overflowed output
        idx_inf = np.logical_not(np.isfinite(lambertwterm))

        # Only re-compute LambertW if it overflowed
        if np.any(idx_inf):
            # Calculate using log(Teta) in case Teta is really big
            Teta2_filtered = (Rsh*Rs*(Iph+Isat)+Rsh*v_points[idx_inf])/(nNsVth*(Rsh+Rs))
            logTeta = np.log(Teta1) + Teta2_filtered
            # Three iterations of Newton-Raphson method to solve
            #  w+log(w)=logTeta. The initial guess is w=logTeta. Where direct
            #  evaluation (above) results in NaN from overflow, 3 iterations
            #  of Newton's method gives approximately 8 digits of precision.
            w = logTeta
            for _ in range(0, 3):
                w = w * (1. - np.log(w) + logTeta) / (1. + w)
            lambertwterm[idx_inf] = w

        i_calc=(Rsh*(Iph+Isat)-v_points)/(Rs+Rsh)-nNsVth/Rs*lambertwterm
        
        return i_calc

    def lambertw_v_from_i(self, Iph, Isat, nNsVth, Rs, Rsh, i_points):
        Vt = self.Vt
        #  is generally more numerically stable
        Gsh = 1.0/Rsh

        Teta1=(Isat*Rsh)/(nNsVth)
        Teta2=(Rsh*Iph+Rsh*Isat-Rsh*i_points)/(nNsVth)

        # overflow is explicitly handled below, so ignore warnings here
        with np.errstate(over='ignore'):
            Teta=Teta1*np.exp(Teta2)

        # may overflow to np.inf
        lambertwterm = lambertw(Teta).real
        
        # Record indices where lambertw input overflowed output
        idx_inf = np.logical_not(np.isfinite(lambertwterm))

        # Only re-compute LambertW if it overflowed
        if np.any(idx_inf):
            # Calculate using log(Teta) in case Teta is really big
            Teta2_filtered = (Iph*Rsh+Isat*Rsh-i_points[idx_inf]*Rsh)/(nNsVth)
            logTeta = np.log(Teta1) + Teta2_filtered
            # Three iterations of Newton-Raphson method to solve
            #  w+log(w)=logTeta. The initial guess is w=logTeta. Where direct
            #  evaluation (above) results in NaN from overflow, 3 iterations
            #  of Newton's method gives approximately 8 digits of precision.
            w = logTeta
            for _ in range(0, 3):
                w = w * (1. - np.log(w) + logTeta) / (1. + w)
            lambertwterm[idx_inf] = w

        v_points = (Iph*Rsh + Isat*Rsh - i_points*Rsh) - i_points*Rs - nNsVth*lambertwterm

        return v_points
    
    def implicit_i_from_v(self, Iph, Isat, nNsVth, Rs, Rsh, v_points):
        
        # Define the function that we need to solve for i_points
        def equation(i_points_guess):
            exp_arg = (v_points + i_points_guess * Rs) / (nNsVth)
            exp_term = np.exp(exp_arg)
            i_calc = Iph - Isat * (exp_term - 1.0) - (v_points + i_points_guess * Rs) / Rsh
            return i_points_guess - i_calc
        
        # Calculate the initial bounds for the current
        i_min = np.zeros_like(v_points)
        i_max = Iph * np.ones_like(v_points)
        
        # Initial mid-point for the binary search
        i_mid = (i_min + i_max) / 2.0
        
        # Perform vectorized binary search
        for _ in range(100):  # Max iterations
            f_mid = equation(i_mid)
            condition = f_mid > 0
            
            # Update search bounds based on the condition
            i_max = np.where(condition, i_mid, i_max)
            i_min = np.where(~condition, i_mid, i_min)
            
            # Recalculate mid-point
            i_mid = (i_min + i_max) / 2.0
            
            # Check if we've converged
            if np.all(np.abs(f_mid) < 1e-4):
                break
        
        return i_mid
    
    def implicit_v_from_i(self, Iph, Isat, nNsVth, Rs, Rsh, i_points):
        Vt = self.Vt
        max_exp_arg = 100  # Limit to avoid overflow
        
        # Define the function that we need to solve
        def equation(v_points_guess):
            exp_arg = (v_points_guess + i_points * Rs) / (nNsVth)
            exp_term = np.exp(exp_arg)
            v_calc = Rsh * Iph - Rsh * Isat * (exp_term - 1.0) - i_points * (Rsh + Rs)
            return v_points_guess - v_calc

        # Calculate the safe voltage range based on the maximum allowed exponent
        v_min = -i_points * Rs - nNsVth * max_exp_arg
        v_max = -i_points * Rs + nNsVth * max_exp_arg
        
        # Initial mid-point for the binary search
        v_mid = (v_min + v_max) / 2.0
        
        # Perform vectorized binary search
        for _ in range(100):  # Max iterations
            f_mid = equation(v_mid)
            condition = f_mid > 0
            
            # Update search bounds based on the condition
            v_max = np.where(condition, v_mid, v_max)
            v_min = np.where(~condition, v_mid, v_min)
            
            # Recalculate mid-point
            v_mid = (v_min + v_max) / 2.0
            
            # Check if we've converged
            if np.all(np.abs(f_mid) < 1e-4):
                break
        
        return v_mid
    
    def formatSol(self, x, error):
        Iph, Isat, nNsVth, Rs, Rsh = x

        dict = {
            "$I_{ph}[A]$": [Iph],
            "$I_s[A]$": [Isat],
            "$eta$" if self.tm is not None  else "$nNsVth$": [nNsVth],
            "$R_s[\Omega]$": [Rs],
            "$R_{sh}[\Omega]$": [Rsh],
            "$Error$": [error]
        }
        
        return pd.DataFrame.from_dict(dict)
    

