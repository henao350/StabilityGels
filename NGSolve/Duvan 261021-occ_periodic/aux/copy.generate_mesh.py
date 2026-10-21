import netgen.geom2d as geom2d
from ngsolve import *
from ngsolve.webgui import Draw

import numpy as np

class gel_bonded2D:
    def __init__(self):
        self.phi0 = 0.2
        self.entropic_unit = 136.6  # measured in MPa
        self.G = 0.13               # measured in MPa
        self.gamma = self.G/self.entropic_unit # 0.0009516837481698391 self.compute_gamma( lamb =1.4874):
        self.lambda_target = 1.99  # 0.13/136.3 = gammafun(1.99)
        self.chi =  0.348 # compute_chi(phi0=0.2035, gamma =0.0009516837481698391, J=J_iso)
        
        self.geometry_bonded()

    def geometry_bonded(self):
        self.L = 90      # measured in mm
        self.d = 1.1    # measured in mm
        self.delta = 0.9
        
        self.geo = geom2d.SplineGeometry()
        
        p1,p2,p3,p4,p5,p6 = [ self.geo.AddPoint (x,y) for x,y in \
                    [(-0.5*self.L,0), (-0.5*self.L*self.delta,0), (0.5*self.L*self.delta, 0), (0.5*self.L,0), \
                        (0.5*self.L,self.d), (-0.5*self.L,self.d)] ]
        self.geo.Append(["line",p1,p2])
        self.geo.Append(["line",p2,p3],bc="bonded_interface")
        self.geo.Append(["line",p3,p4])
        self.geo.Append(["line",p4,p5])
        self.geo.Append(["line",p5,p6])
        self.geo.Append(["line",p6,p1])

    def phi(self, J):
        return self.phi0/J

    def H(self, J):
        return (J - self.phi0)*log(1-self.phi(J))  + self.phi0 * self.chi*(1-self.phi(J))

    def dH(self, J):
        return self.phi(J) + np.log(1-self.phi(J)) + self.chi * self.phi(J)**2
    
    # compute gamma using uniaxial approximation
    def gammafun(self,lamb):
        return -self.dH(lamb)/lamb
        
    # energy density
    def W(self, F):
        J = Det(F)
        C = F.trans* F
        return 0.5*self.gamma * Trace(C) + self.H(J)
        #return 0.5*self.gamma * Trace(C) + (J-1)**2