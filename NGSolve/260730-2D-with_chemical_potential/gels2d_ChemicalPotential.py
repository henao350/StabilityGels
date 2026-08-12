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
        # self.mu_bar = mu_bar
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
        mu_bar = gel.mu_bar.Get()
        # nu=gel.entropic_unit
        
        # reference_energy_density =  self.reference_energy_density        
        # return 0.5*G*(First_principal_invariant - 3 -2*log(J)) + nu*gel.H(J) - reference_energy_density

        # [Kang & Huang JMPS 2010, Eqs. (2.6), (2.7), (2.9) and (3.5)]
        elastic = 0.5*G*(First_principal_invariant - 3 -2*log(J))
        Flory_Huggins = self.entropic_unit*gel.H(J)
        chemical_potential = - self.entropic_unit * mu_bar * (J - phi0)
        pressure_external_solvent = (J - phi0)*p
        return elastic + Flory_Huggins + chemical_potential + pressure_external_solvent

class Solve_gel_bonded:
    def __init__(self, gel, order=3):           # corner_refinement capability not yet activated
        self.gel = gel
        self.order = order

    def add_mesh(self, mesh_file):
        self.mesh = Mesh(mesh_file)

    def Space(self):
        # Finite element space with zero-displacement boundary condition on the whole interface
        self.fes = VectorH1(self.mesh, order=self.order, \
                            dirichlet="bonded_interface|debonded_interface")
        print('nDoF = {}'.format(self.fes.ndof))

    def model(self):
        u = self.fes.TrialFunction()
        I = Id(self.mesh.dim)
        F = I + Grad(u)

        def negpart(var):
            #return max(-var,0)
            return (sqrt(var**2)-var)*0.5        

        AA = 1e5
        # hydrogel model        
        self.a = BilinearForm(self.fes, symmetric=False)
        self.a += Variation(  self.gel.W(F).Compile() * dx)
        self.a += Variation(  AA*negpart(y+u[1])**2 * dx)


    def Solve_incremental_softening(self):
        self.Space()        
        self.gfu = GridFunction(self.fes)
        self.gfu.vec[:] = 0

        mu_bar_end = self.gel.mu_bar.Get()
        print(f'mu_bar_end = {mu_bar_end}')
        lambda_initial = 1.1
        mu_bar_0 = self.gel.mu_fun(lambda_initial)
        # nIterations = 15
        nIterations = 1
        self.gel.mu_bar.Set(mu_bar_0)
        self.model()

        #From 0 to nIterations-1
        lambda_list = np.linspace(lambda_initial, self.gel.lambda_target, nIterations, endpoint = False)
        #gamma_list = [self.gel.gammafun(la) for la in lambda_list]
        mu_bar_list = [self.gel.mu_fun(la) for la in lambda_list]
        # final iteration
        mu_bar_list.append(mu_bar_end)

        filename = 'gridfunctions/result_gelsMu2D'+self.gel.filename_suffix + f'_order={self.order}'
        print(f'filename={filename}')    

        tol=1e-3; maxits=100;

        indexes_iterations = range(nIterations+1)
        for numIteration in indexes_iterations:
            mu_bar_i = mu_bar_list [numIteration]
            print(f"*** Iteration #{numIteration}, mu_bar = {mu_bar_i}")

            if numIteration==nIterations:
                tol=1e-6; maxits=500;
            
            self.gel.mu_bar.Set(mu_bar_i)
            self.gfu, _,_ = SolveNonlinearMinProblem(a= self.a, gfu = self.gfu,\
                        FreeDofs =self.fes.FreeDofs(), maxits=maxits, tol=tol)

            # Draw(self.gfu, deformation=True)            
            # self.gfu.Save(filename + '_iter=' + str(numIteration).zfill(2) + '.gfu')
            self.gfu.Save(filename + '.gfu')


def SolveNonlinearMinProblem(a, gfu, FreeDofs, tol=1e-08, maxits=50, alpha=5e-2):#, scenes=None):
    res = gfu.vec.CreateVector()
    du  = gfu.vec.CreateVector()

    for it in range(maxits):
        #print ("Newton iteration {:3}".format(it),end=", ")
        # print ("energy = {:16}".format(a.Energy(gfu.vec)),end="")

        # solve linearized problem:
        a.Apply (gfu.vec, res)
        a.AssembleLinearization (gfu.vec)
        inv = a.mat.Inverse(FreeDofs)
        #alpha = 5e-2
        du.data = - alpha * inv * res

        #update iteration
        gfu.vec.data += du

        #stopping criteria
        stopcritval = sqrt(abs(InnerProduct(du,res)))
        #print ("<A u",it,", A u",it,">_{-1}^0.5 = ", stopcritval)
        if stopcritval < tol:
            
            break

        #for sc in scenes:
        #    sc.Redraw()

    return gfu, stopcritval, it

if __name__ == '__main__':
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
    #
    modelling = Solve_gel_bonded (gel, order=order)
    mesh_file = 'meshes1_62/mesh51.vol'
    modelling.add_mesh(mesh_file)
    modelling.Solve_incremental_softening()    
    print("Time elapsed =" + str(datetime.datetime.now() - start_time))
    # 

