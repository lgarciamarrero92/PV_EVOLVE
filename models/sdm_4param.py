import numpy as np
from scipy.special import lambertw
from core.params import Iph, Isat, eta, Rs, Rsh, nNsVth
from core.iv import IVModel
from utils.constants import BOLTZMANN_CONSTANT, ELECTRON_VOLT
from utils.helpers import celsius_to_kelvin

class SDM4PARAM(IVModel):
    def __init__(self,tm=None,ns=None,method="v_from_i",bounds={}):
        self.name = "SDM4PARAM"
        self.method = method
        self.bounds = bounds
        self.Vt = None
        
        if tm is not None and ns is not None:
            #Calculus of Vt (Thermal voltage)
            T=celsius_to_kelvin(tm)
            self.Vt=(ns*BOLTZMANN_CONSTANT*T)/ELECTRON_VOLT

            self.params = [
                Iph(),
                Isat(),
                eta(),
                Rs()
            ]

        else:
            self.params = [
                Iph(),
                Isat(),
                nNsVth(),
                Rs()
            ]

        super().__init__(params=self.params)

    def _calc(self,params,v_points=None,i_points=None):
        if self.Vt is not None:
            Iph, Isat, eta, Rs = params
            nNsVth = self.Vt*eta
        else:
            Iph, Isat, nNsVth, Rs = params

        if self.method == 'v_from_i':
            v_calc = self.v_from_i(Iph, Isat, nNsVth, Rs, i_points)
            return (v_calc,i_points)
        
        else:
            raise NotImplementedError("Method not implemented")
    
    def v_from_i(self, Iph, Isat, nNsVth, Rs, i_points):
        arg = ((Iph - i_points) / Isat) + 1
        
        # Create an array for the results initialized with np.nan
        v_points = np.full_like(i_points, np.nan, dtype=float)
        
        # Determine the valid mask where arg > 0
        valid_mask = arg > 0
        
        # Calculate only for the valid entries
        v_points[valid_mask] = nNsVth * np.log(arg[valid_mask]) - i_points[valid_mask] * Rs

        return v_points
    

