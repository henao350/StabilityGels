# ClassGels3.py
# Library that contains the Newton solver SolveNonlinearMinProblem,
#  the class gel2d, which implements the 2D hydrogel model
#  and the class Solve_gel_bonded, which implements the finite element model for a bonded hydrogel
#  This version, based on ClassGel2d.py, is intended to be used with gels3.py, which implements a different approach to the solution of the problem, based on the target theoretical uniaxial stretch lambda_h instead of the chemical potenti
from ngsolve import *
import numpy as np
import scipy.optimize

def SolveNonlinearMinProblem(a, gfu, FreeDofs, tol=1e-08, maxits=50, alpha=5e-2):#, scenes=None):
    res = gfu.vec.CreateVector()
    du  = gfu.vec.CreateVector()
    for it in range(maxits):
        # print ("Newton iteration {:3}".format(it),end=", ")
        # print ("mu_bar = {:16}".format(a.mu_bar.Get()),end=", ")
        # print ("energy = {:16}".format(a.Energy(gfu.vec)),end="\n")
        # solve linearized problem:
        a.Apply (gfu.vec, res)
        a.AssembleLinearization (gfu.vec)
        inv = a.mat.Inverse(FreeDofs)
        du.data = - alpha * inv * res
        #update iteration
        gfu.vec.data += du
        #stopping criteria
        stopcritval = sqrt(abs(InnerProduct(du,res)))
        # print ("<A u",it,", A u",it,">_{-1}^0.5 = ", stopcritval)
        if stopcritval < tol:
            break
        #for sc in scenes:
        #    sc.Redraw()
    return gfu, stopcritval, it

class gel2d:
    def __init__(
        self,
        length=100.0,
        thickness=20.0,
        phi0=0.2,
        lambda_target=2.0
    ):
        self.length = length
        self.d = thickness
        self.phi0 = phi0
        # self.mu_bar = mu_bar
        T = 25 + 273.15     # in [K]
        K_B = 1.380649e-23  # in [J·K^{-1}]
        V_m = 3e-29         # in [m^3]
        self.entropic_unit = K_B*T/V_m*1e-6     # in [MPa]        
        self.gamma = 1e-3
        G_value = self.gamma*self.entropic_unit    # in [MPa]
        self.G = Parameter(G_value)  # in [MPa]
        self.chi = 0.4
        vapor_pressure = 3.2e-3   # in [MPa]
        self.p0_bar = vapor_pressure/self.entropic_unit
        #
        self.lambda_target = lambda_target
        mu_bar = self.mu_fun(lambda_target)
        print(f'Computed mu_bar = {mu_bar} for lambda_target = {lambda_target}')
        self.mu_bar = Parameter(mu_bar)

    def phi(self, J):
        return self.phi0/J

    def H(self, J): # Flory-Huggins energy density in Lagrangian coordinates (energy per reference volume)
        return (J - self.phi0)*log(1-self.phi(J))  + self.phi0 * self.chi*(1-self.phi(J))

    def dH(self, J):
        phi = self.phi(J)
        chi = self.chi
        # return self.phi(J) + np.log(1-self.phi(J)) + self.chi * self.phi(J)**2
        return phi + np.log(1-phi) + chi*phi**2

    # Computes gamma (i.e. G/entropic_unit) from a desired target value lamb for lambda_uniaxial
    # and a given mu_bar and p_bar, using [Kang & Huang JMPS 2010, Eq.(3.6)]
    def Gfun(self, lamb):
        mu_bar = self.mu_bar.Get()
        p_bar = self.p0_bar*np.exp(mu_bar)
        return ((mu_bar - p_bar -self.dH(lamb))/(lamb-1/lamb))*self.entropic_unit

    #### Compute nondimensionalized chemical potencial mu_bar
    #### for given lambda, phi0, chi and gamma,
    #### using [Kang & Huang JMPS 2010, Eq.(3.6)]
    #### and the approximation p_bar = p0_bar
    def mu_fun(self, lamb):
        p0_bar = self.p0_bar
        p_bar = p0_bar  # it is an approximation
        gamma = self.gamma
        return self.dH(lamb) + gamma*(lamb - 1/lamb) + p_bar   # [Kang & Huang JMPS 2010, Eq.(3.6), but with p replaced by p_0]    

