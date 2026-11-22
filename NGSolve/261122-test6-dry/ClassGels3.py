# ClassGels3.py
# Library that contains the Newton solver SolveNonlinearMinProblem,
#  the class gel2d, which implements the 2D hydrogel model
#  and the class Solve_gel_bonded, which implements the finite element model for a bonded hydrogel
#  This version, based on ClassGel2d.py, is intended to be used with gels3.py, which implements a different approach to the solution of the problem, based on the target theoretical uniaxial stretch lambda_h instead of the chemical potenti
from ngsolve import *
import numpy as np
# import scipy.optimize
import logging

def SolveNonlinearMinProblem(
        a, 
        gfu,
        FreeDofs,
        tol=1e-08,
        maxits=50,
        alpha=5e-2,        
        Save_iterations = False,
        filename_main = 'test2026'
        ):#, scenes=None):
    # ---------------------------------------------------------
    # SETUP FOR LOGGING
    logger = logging.getLogger(__name__)    
    # ---------------------------------------------------------
    # NEWTON'S INITIAL GUESS
    # Saves current gfu as Newton iteration 0
    it = 0 
    if Save_iterations:
        filename = filename_main + '_Newton=' + str(it).zfill(2)
        gfu.Save(filename + '.gfu')
        logger.info (f'Newton iteration {it:3}')
        # print ("mu_bar = {:16}".format(a.mu_bar.Get()),end=", ")
        logger.info (f'a.Energy = {a.Energy(gfu.vec):16}')
    # ---------------------------------------------------------
    # NEWTON'S METHOD ITSELF
    res = gfu.vec.CreateVector()
    du  = gfu.vec.CreateVector()
    mesh = gfu.space.mesh
    for it in range(1,maxits+1):
        # solve linearized problem:
        a.Apply (gfu.vec, res)
        a.AssembleLinearization (gfu.vec)
        inv = a.mat.Inverse(FreeDofs)
        du.data = - alpha * inv * res
        #update iteration
        gfu.vec.data += du
        #
        if Save_iterations:
        # if Save_iterations and (it % 50 == 0 or it == maxits):
            filename = filename_main + '_Newton=' + str(it).zfill(2)
            gfu.Save(filename + '.gfu')
        #stopping criteria
        stopcritval = sqrt(abs(InnerProduct(du,res)))
        logger.info (f'Newton iteration {it:3}')
        logger.info (f'a.Energy = {a.Energy(gfu.vec):16}')
        # print ("<A u",it,", A u",it,">_{-1}^0.5 = ", stopcritval)
        logger.info (f'<A u{it}, A u{it}>_{-1}^0.5 ?=? <du, res>_{2}^0.5 = {stopcritval}')
        vertex_values = [(y+gfu[1])(mesh(*p)) for p in mesh.ngmesh.Points()]
        logger.info(f'Maximum swollen gel thickness: {max(vertex_values)}') 
        if stopcritval < tol:
            break
        #for sc in scenes:
        #    sc.Redraw()
    if stopcritval < tol:
        logger.info(f'Newton\'s method arrived to specified tolerance after {it} iterations')
    else:
        logger.info(f'Newthon\'s method did not arrive to the specified tolerance during the {maxits} iterations')
    return gfu, stopcritval, it

class gel2d:
    def __init__(
        self,
        length=100.0,
        thickness=20.0,
        phi0=0.2,
        lambda_target=2.0
    ):
        # ---------------------------------------------------------
        # SETUP FOR LOGGING
        logger = logging.getLogger(__name__)    
        # ---------------------------------------------------------
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
        # mu_bar = round(mu_bar, 3)  # round to 3 decimal places
        logger.info(f'Computed mu_bar = {mu_bar} for lambda_target = {lambda_target}')
        self.mu_bar = Parameter(mu_bar)
        self.p_bar = self.p0_bar*exp(self.mu_bar)  # [Kang & Huang JMPS 2010, Eq. (2.3)]        
        #
        logger.info(f'Entropic unit = {self.entropic_unit:.2f} [MPa], G = {self.G.Get():.4f} [MPa], ' \
                + f'chi = {self.chi}, p0_bar = {self.p0_bar:.7f}, lambda_target = {self.lambda_target}')
        logger.info(f'gel.Gfun(lambda_target) = {self.Gfun(self.lambda_target)}')

        self.filename_suffix = self.filename_suffix_()
        logger.info(f'filename_suffix={self.filename_suffix}')

    def phi(self, J):
        return self.phi0/J

    def H(self, J): # Flory-Huggins energy density in Lagrangian coordinates (energy per reference volume)
        return (J - self.phi0)*log(1-self.phi(J))  + self.phi0 * self.chi*(1-self.phi(J))

    def dH(self, J):
        phi = self.phi(J)
        chi = self.chi
        # return self.phi(J) + np.log(1-self.phi(J)) + self.chi * self.phi(J)**2
        return phi + np.log(1-phi) + chi*phi**2

    def dH_fes(self, J):
        phi = self.phi(J)
        chi = self.chi
        # return self.phi(J) + np.log(1-self.phi(J)) + self.chi * self.phi(J)**2
        return phi + log(1-phi) + chi*phi**2

    # Computes gamma (i.e. G/entropic_unit) from a desired target value lamb for lambda_uniaxial
    # and a given mu_bar and p_bar, using [Kang & Huang JMPS 2010, Eq.(3.6)]
    def Gfun(self, lamb):
        mu_bar = self.mu_bar.Get()
        p_bar = self.p0_bar*np.exp(mu_bar)
        return ((mu_bar - p_bar -self.dH(lamb))/(lamb-1/lamb))*self.entropic_unit

    # def find_lambda_target(self):
    #     G_value = self.G.Get()
    #     def auxFunctionLambdaTargetShearModulus(lamb):
    #         return self.Gfun(lamb) - G_value
    #     # Among 100 trials, chooses the value of lambda
    #     # for which Gfun is as the closest to the target G_value
    #     lambda_start = 1.0001; lambda_end = 10; numPoints=1000;
    #     lambda_values = np.linspace(lambda_start, lambda_end, numPoints)        
    #     lambda_initial_fsolve = lambda_start
    #     best_abs_dist= abs(auxFunctionLambdaTargetShearModulus(lambda_start))
    #     for lamb in lambda_values:
    #         dist = auxFunctionLambdaTargetShearModulus(lamb)
    #         if abs(dist) < best_abs_dist:
    #             best_abs_dist = abs(dist)
    #             lambda_initial_fsolve = lamb   
    #     # and finally, for higher precision, fsolve is called
    #     # This seems to be necessary since the functin Gfun is
    #     # extremely sensitive to changes in lambda even in its fourth decimal place
    #     lambda_target = (scipy.optimize.fsolve(auxFunctionLambdaTargetShearModulus, lambda_initial_fsolve))[0]   # approximate solution to  self.G_fun(lamb) = G_value, for given G_value          
    #     return lambda_target

    # def find_lambda_for_mu(self):
    #     mu_value = gel.mu_bar.Get()
    #     def auxFunctionLambdaMu(lamb):
    #         return gel.mu_fun(lamb) - mu_value
    #     # Among 100 trials, chooses the value of lambda
    #     # for which mu_fun is closest to the target mu_value
    #     lambda_start = 1.0001; lambda_end = 10; numPoints=1000;
    #     lambda_values = np.linspace(lambda_start, lambda_end, numPoints)        
    #     lambda_initial_fsolve = lambda_start
    #     best_abs_dist= abs(auxFunctionLambdaMu(lambda_start))
    #     for lamb in lambda_values:
    #         dist = auxFunctionLambdaMu(lamb)
    #         if abs(dist) < best_abs_dist:
    #             best_abs_dist = abs(dist)
    #             lambda_initial_fsolve = lamb   
    #     # and finally, for higher precision, fsolve is called
    #     # This seems to be necessary since the function mu_fun is potentially 
    #     # extremely sensitive to changes in lambda
    #     lamb = (scipy.optimize.fsolve(auxFunctionLambdaMu, lambda_initial_fsolve))[0]   # approximate solution to  self.mu_fun(lamb) = mu_value, for given mu_value          

    #### Compute nondimensionalized chemical potencial mu_bar
    #### for given lambda, phi0, chi and gamma,
    #### using [Kang & Huang JMPS 2010, Eq.(3.6)]
    #### and the approximation p_bar = p0_bar
    def mu_fun(self, lamb):
        p0_bar = self.p0_bar
        p_bar = p0_bar  # it is an approximation
        gamma = self.gamma
        return self.dH(lamb) + gamma*(lamb - 1/lamb) + p_bar   # [Kang & Huang JMPS 2010, Eq.(3.6), but with p replaced by p_0]    

    # energy density in [MPa]
    def W(self, F):      
        J = Det(F)
        C = F.trans* F
        First_principal_invariant = 1.0 + Trace(C)
        #
        gel = self
        G = gel.G
        phi0 = gel.phi0
        p_bar = gel.p_bar
        p = p_bar * self.entropic_unit
        mu_bar = gel.mu_bar.Get()
        #
        # nu=gel.entropic_unit       
        # reference_energy_density =  self.reference_energy_density        
        # return 0.5*G*(First_principal_invariant - 3 -2*log(J)) + nu*gel.H(J) - reference_energy_density
        #
        # [Kang & Huang JMPS 2010, Eqs. (2.6), (2.7), (2.9) and (3.5)]
        elastic = 0.5*G*(First_principal_invariant - 3 -2*log(J))
        Flory_Huggins = self.entropic_unit*gel.H(J)
        # - mu * C = - mu * (J-phi0)/V_m = - (mu_bar*K_B*T)*(J-phi0)/V_m = - entropic_unit*mu_bar*(J-phi0)
        chemical_potential = - self.entropic_unit * mu_bar * (J - phi0)
        pressure_external_solvent = (J - phi0)*p
        return elastic + Flory_Huggins + chemical_potential + pressure_external_solvent

    # Horizontal stress
    def sigma_xx(self, F):
        J = Det(F)
        #
        gel = self
        G = gel.G
        p_bar = gel.p_bar
        p = p_bar * self.entropic_unit
        mu_bar = gel.mu_bar.Get()
        #
        elastic = (G/J)*(F[0,0]**2 + F[0,1]**2) - G/J
        Flory_Huggins = self.entropic_unit*gel.dH_fes(J)
        chemical_potential = - self.entropic_unit * mu_bar
        pressure_external_solvent = p
        return elastic + Flory_Huggins + chemical_potential + pressure_external_solvent

    def filename_suffix_(self):
        phi0=self.phi0; mu_bar=self.mu_bar.Get()
        return f"_phi0={phi0}_absMuBar={abs(mu_bar):.3f}"

    def filename_main(self, simulation_name):
        return f'{simulation_name}/result_gelsMu2D' + self.filename_suffix
    
    # def filename_Newton_it(self, simulation_name, it):
    #     return self.filename_main(simulation_name) + '_Newton=' + str(it).zfill(2)

class Solve_gel_bonded:
    def __init__(self, gel, order=3):           # corner_refinement capability not yet activated
        self.gel = gel
        self.order = order

    def update_chemical_potential(self, new_mu_bar):
        # ---------------------------------------------------------
        # SETUP FOR LOGGING
        logger = logging.getLogger(__name__)    
        # ---------------------------------------------------------
        solver=self
        solver.gel.mu_bar.Set(new_mu_bar)
        solver.model()
        solver.gel.filename_suffix = solver.gel.filename_suffix_()
        logger.info(f'mu_bar is now {solver.gel.mu_bar.Get()} and filename_suffix={solver.gel.filename_suffix}')    

    def add_mesh(self, mesh_file):
        # ---------------------------------------------------------
        # SETUP FOR LOGGING
        logger = logging.getLogger(__name__)    
        # ---------------------------------------------------------
        logger.info(f'mesh_file={mesh_file}')
        self.mesh = Mesh(mesh_file)
        logger.info(f'nElements = {self.mesh.ne}, nVertices = {self.mesh.nv}')

    def Space(self):
        # ---------------------------------------------------------
        # SETUP FOR LOGGING
        logger = logging.getLogger(__name__)    
        # ---------------------------------------------------------
        # # Finite element space with zero-displacement boundary condition on the whole interface
        # # and periodic boundary conditions identifying left and right.
        # self.fes = VectorH1(self.mesh, order=self.order, \
        #                     dirichlet="bot", dirichletx="left|right")
        # print(f'sum(fes.FreeDofs()) = {sum(self.fes.FreeDofs())}')
        ###
        # Finite element space with zero-displacement boundary condition on the whole interface
        # and periodic boundary conditions identifying left and right.
        self.fes = Periodic(VectorH1(self.mesh, order=self.order, \
                            dirichlet="bot"))
        logger.info(f'sum(fes.FreeDofs()) = {sum(self.fes.FreeDofs())}')
        ###
        # # Finite element space with zero-displacement boundary condition on the whole interface
        # self.fes = VectorH1(self.mesh, order=self.order, \
        #                     dirichlet="bonded_interface|debonded_interface")
        # print('nDoF = {}'.format(self.fes.ndof))

    def model(self):
        u = self.fes.TrialFunction()
        I = Id(self.mesh.dim)
        F = I + Grad(u)
        def negpart(var):
            #return max(-var,0)
            return (sqrt(var**2)-var)*0.5        
        # hydrogel model        
        AA = 1e5
        self.a = BilinearForm(self.fes, symmetric=False)
        self.a += Variation(  self.gel.W(F).Compile() * dx)
        self.a += Variation(  AA*negpart(y+u[1])**2 * dx)
