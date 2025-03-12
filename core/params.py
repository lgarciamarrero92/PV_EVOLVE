from pymoo.core.variable import Real, Integer, Choice, Binary
import numpy as np

class Param:
    def __init__(self, name, bounds = None, param_type="real", latex_name=None, unit = None):
        self.name = name
        self.latex_name = latex_name
        self.unit = unit
        self.bounds = bounds
        self.param_type = param_type
        if param_type == "real":
           self.var = Real(bounds=bounds)
        elif param_type == "integer":
            self.var = Integer(bounds=bounds)
           
class Ns(Param):
    def __init__(self, instance = None, bounds = None):
        name = "Ns" if (instance is None) else "Ns"+str(instance)
        latex_name = "$Ns$" if (instance is None) else "$Ns_{"+str(instance)+"}$"
        unit = "-"
        if bounds is None:
            bounds = (1,2)
        super().__init__(name, bounds = bounds, param_type="integer", latex_name=latex_name, unit=unit)

class Iph(Param):
    def __init__(self, instance = None, bounds = None):
        name = "Iph" if (instance is None) else "Iph"+str(instance)
        latex_name = "$I_{ph}$" if (instance is None) else "$I{ph"+str(instance)+"}$"
        unit = "A"
        if bounds is None:
            bounds = (0.1,12)
        super().__init__(name, bounds = bounds, latex_name=latex_name, unit=unit)

class Isat(Param):
    def __init__(self, instance = None, bounds = None):
        name = "Isat" if (instance is None) else "Isat"+str(instance)
        latex_name = "$I_{sat}$" if (instance is None) else "$I{sat"+str(instance)+"}$"
        unit = "A"
        if bounds is None:
            bounds = (0,1e-3)
        super().__init__(name, bounds = bounds, latex_name=latex_name, unit=unit)

class eta(Param):
    def __init__(self, instance = None, bounds = None):
        name = "eta" if (instance is None) else "eta"+str(instance)
        latex_name = "$eta$" if (instance is None) else "$eta{"+str(instance)+"}$"
        unit = "-"
        if bounds is None:
            bounds = (1,2)
        super().__init__(name, bounds = bounds, latex_name=latex_name, unit=unit)

class nNsVth(Param):
    def __init__(self, instance = None, bounds = None):
        name = "nNsVth" if (instance is None) else "nNsVth"+str(instance)
        latex_name = "$nNsVth$" if (instance is None) else "$nNsVth{"+str(instance)+"}$"
        unit = "V"
        if bounds is None:
            bounds = (0,10)
        super().__init__(name, bounds = bounds, latex_name=latex_name, unit=unit)


class Rs(Param):
    def __init__(self, instance = None, bounds = None):
        name = "Rs" if (instance is None) else "Rs"+str(instance)
        latex_name = "$R_s$" if (instance is None) else "$R_{s"+str(instance)+"}$"
        unit = "Ohm"
        if bounds is None:
            bounds = (0,5)
        super().__init__(name, bounds = bounds, latex_name=latex_name, unit=unit)

class Rsh(Param):
    def __init__(self, instance = None, bounds = None):
        name = "Rsh" if (instance is None) else "Rsh"+str(instance)
        latex_name = "$R_{sh}$" if (instance is None) else "$R_{sh"+str(instance)+"}$"
        unit = "Ohm"
        if bounds is None:
            bounds = (0,1e4)
        super().__init__(name, bounds = bounds, latex_name=latex_name, unit=unit)