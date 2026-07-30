from ngsolve import *

import numpy as np
import scipy.optimize
import csv
import datetime
import math

class gel2d:
    def __init__(
        self,
        length=90.0,
        thickness=1.62,
        phi0=0.2,
        mu_bar=-0.02
    ):
        self.length = length
        self.d = thickness
        self.phi0 = phi0
        self.mu_bar = mu_bar
        T = 25 + 273.15     # in [K]
        K_B = 1.380649e-23  # in [J·K^{-1}]
        V_m = 3e-29         # in [m^3]
        self.entropic_unit = K_B*T/V_m*1e-6     # in [MPa]        
        self.gamma = 1e-3
        self.G = self.gamma*self.entropic_unit  # in [MPa]
        self.chi = 0.4
        vapor_pressure = 3.2e-3   # in [MPa]
        self.p0_bar = vapor_pressure/self.entropic_unit
        self.mu_bar = Parameter(mu_bar)
        self.p_bar = Parameter(self.p0_bar*np.exp(mu_bar))  # [Kang & Huang JMPS 2010, Eq. (2.3)]

        def auxFunctionLambdaTarget(lamb):
            return self.mu_fun(lamb) - mu_bar

        self.lambda_target = (scipy.optimize.fsolve(auxFunctionLambdaTarget, 1.5))[0]   # approximate solution to  self.mu_fun(lamb) = mu_bar, for given mu_bar
        self.lambda_target = math.floor(100*self.lambda_target)/100
        # self.lambda_target = 1.46   
        print(f'Entropic unit = {self.entropic_unit:.2f} [MPa], G = {self.G:.4f} [MPa], ' \
                + f'chi = {self.chi}, p0_bar = {self.p0_bar:.7f}, lambda_target = {self.lambda_target}, ' \
                + f'gel.mu_fun(lambda_target) = {self.mu_fun(self.lambda_target)}')

        self.filename_suffix = f"_phi0={phi0}_absMuBar={abs(mu_bar):.3f}"
        print(f'filename_suffix='+ self.filename_suffix)

    def phi(self, J):
        return self.phi0/J

    #### Compute nondimensionalized chemical potencial mu_bar
    def mu_fun(self, lamb):
        p0_bar = self.p0_bar
        p_bar = p0_bar  # it is an approximation
        phi = self.phi(lamb)   # phi = phi0/lamb. When phi0=1, phi=1/lamb
        chi = self.chi
        gamma = self.gamma
        return phi + np.log(1-phi) + chi*phi**2 + gamma*(lamb - 1/lamb) + p_bar   # [Kang & Huang JMPS 2010, Eq.(3.6), but with p replaced by p_0]

    def H(self, J): # Flory-Huggins energy density in Lagrangian coordinates (energy per reference volume)
        return (J - self.phi0)*log(1-self.phi(J))  + self.phi0 * self.chi*(1-self.phi(J))

    # energy density in [MPa]
    def W(self, F):
               
        J = Det(F)
        C = F.trans* F
        First_principal_invariant = 1.0 + Trace(C)
        
        gel = self
        G = gel.G
        phi0 = gel.phi0
        p_bar = gel.p_bar
        p = p_bar * self.entropic_unit
        # nu=gel.entropic_unit
        
        # reference_energy_density =  self.reference_energy_density        
        # return 0.5*G*(First_principal_invariant - 3 -2*log(J)) + nu*gel.H(J) - reference_energy_density

        # [Kang & Huang JMPS 2010, Eqs. (2.6), (2.7), (2.9) and (3.5)]
        elastic = 0.5*G*(First_principal_invariant - 3 -2*log(J))
        Flory_Huggins = self.entropic_unit*gel.H(J)
        chemical_potential = - self.entropic_unit * mu_bar * (J - phi0)
        pressure_external_solvent = (J - phi0)*p
        return elastic + Flory_Huggins + chemical_potential + pressure_external_solvent


if __name__ == '__main__':
    print("Hello")
    start_time = datetime.datetime.now()
    #### READ PARAMETERS ####
    # It will be assumed that the arguments passed from shell
    # are, in that order:
    # the length; the thickness; the order for the finite element space;
    # the initial polymer volume fraction phi0; 
    # the absolute value of the nondimensionalized chemical potential
    #    mu_bar = mu / (K_B*T)
    # (which, in turn, will be assumed to be negative)
    # For example,
    #    90 1.62 3 0.2 0.5
    # means L=90, d=1.62, order=3, phi0=0.2, mu_bar=-0.5
    data = sys.argv
    L = float(data[1])
    d = float(data[2])
    order = int(data[3])
    phi0 = float(data[4])
    mu_input = float(data[5])    
    mu_bar = -mu_input  # internal sign convention
    print(f'L={L} [mm], d={d} [mm], order={order}, phi0={phi0}, mu_bar={mu_bar}')
    #
    gel = gel2d(L,d,phi0,mu_bar)

