# ClassGel2d.py
# Library that contains the Newton solver SolveNonlinearMinProblem with Backtracking Line Search,
# the class gel2d, which implements the 2D hydrogel model,
# and the class Solve_gel_bonded, which implements the finite element model for a bonded hydrogel.
from ngsolve import *
import numpy as np
import scipy.optimize

def SolveNonlinearMinProblem(a, gfu, FreeDofs, tol=1e-06, maxits=100, alpha_init=1.0):
    """
    Newton solver with backtracking line search to guarantee energy stability
    and prevent numerical breakdown in the Flory-Huggins logarithmic term (ln(1 - phi)).
    """
    res = gfu.vec.CreateVector()
    du  = gfu.vec.CreateVector()
    temp_u = gfu.vec.CreateVector()
    
    stopcritval = 0.0
    it = 0
    
    for it in range(maxits):
        # Apply non-linear form to compute residual
        a.Apply(gfu.vec, res)
        stopcritval = sqrt(abs(InnerProduct(res, res)))
        
        if stopcritval < tol:
            break

        # Assemble tangent matrix
        a.AssembleLinearization(gfu.vec)
        inv = a.mat.Inverse(FreeDofs)
        
        # Newton direction
        du.data = - inv * res
        
        # -----------------------------------------------------
        # Backtracking Line Search
        # -----------------------------------------------------
        alpha = alpha_init
        min_alpha = 1e-5
        
        try:
            energy_old = a.Energy(gfu.vec)
        except:
            energy_old = float('inf')

        while alpha > min_alpha:
            temp_u.data = gfu.vec + alpha * du
            try:
                energy_new = a.Energy(temp_u)
                if np.isnan(energy_new) or np.isinf(energy_new) or energy_new > energy_old:
                    alpha *= 0.5
                else:
                    break
            except:
                alpha *= 0.5
        
        # Update solution vector
        gfu.vec.data += alpha * du

    return gfu, stopcritval, it


class gel2d:
    def __init__(
        self,
        length=100.0,
        thickness=20.0,
        phi0=0.2,
        mu_bar=-0.02
    ):
        self.length = length
        self.d = thickness
        self.phi0 = phi0
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
        self.mu_bar = Parameter(mu_bar)
        self.p_bar = self.p0_bar*exp(self.mu_bar)  # [Kang & Huang JMPS 2010, Eq. (2.3)]        

        self.lambda_target = self.find_lambda_target()

        print(f'Entropic unit = {self.entropic_unit:.2f} [MPa], G = {self.G.Get():.4f} [MPa], ' \
                + f'chi = {self.chi}, p0_bar = {self.p0_bar:.7f}, lambda_target = {self.lambda_target}')

        self.filename_suffix = f"_phi0={phi0}_absMuBar={abs(mu_bar):.3f}"

    def phi(self, J):
        return self.phi0/J

    def H(self, J): # Flory-Huggins energy density
        return (J - self.phi0)*log(1-self.phi(J)) + self.phi0 * self.chi*(1-self.phi(J))

    def dH(self, J):
        phi = self.phi(J)
        chi = self.chi
        return phi + np.log(1-phi) + chi*phi**2

    def Gfun(self, lamb):
        mu_bar = self.mu_bar.Get()
        p_bar = self.p0_bar*np.exp(mu_bar)
        return ((mu_bar - p_bar - self.dH(lamb))/(lamb - 1/lamb))*self.entropic_unit

    def find_lambda_target(self):
        G_value = self.G.Get()
        def auxFunctionLambdaTargetShearModulus(lamb):
            return self.Gfun(lamb) - G_value

        lambda_start = 1.0001; lambda_end = 10; numPoints = 1000
        lambda_values = np.linspace(lambda_start, lambda_end, numPoints)        
        lambda_initial_fsolve = lambda_start
        best_abs_dist = abs(auxFunctionLambdaTargetShearModulus(lambda_start))
        for lamb in lambda_values:
            dist = auxFunctionLambdaTargetShearModulus(lamb)
            if abs(dist) < best_abs_dist:
                best_abs_dist = abs(dist)
                lambda_initial_fsolve = lamb   
        
        lambda_target = (scipy.optimize.fsolve(auxFunctionLambdaTargetShearModulus, lambda_initial_fsolve))[0]          
        return lambda_target

    def mu_fun(self, lamb):
        p0_bar = self.p0_bar
        p_bar = p0_bar
        gamma = self.gamma
        return self.dH(lamb) + gamma*(lamb - 1/lamb) + p_bar

    def W(self, F):      
        J = Det(F)
        C = F.trans * F
        First_principal_invariant = 1.0 + Trace(C)
        
        G = self.G
        phi0 = self.phi0
        p_bar = self.p_bar
        p = p_bar * self.entropic_unit
        mu_bar = self.mu_bar.Get()

        elastic = 0.5*G*(First_principal_invariant - 3 - 2*log(J))
        Flory_Huggins = self.entropic_unit*self.H(J)
        chemical_potential = - self.entropic_unit * mu_bar * (J - phi0)
        pressure_external_solvent = (J - phi0)*p
        return elastic + Flory_Huggins + chemical_potential + pressure_external_solvent


class Solve_gel_bonded:
    def __init__(self, gel, order=3):
        self.gel = gel
        self.order = order

    def add_mesh(self, mesh_file):
        print(f'mesh_file={mesh_file}')
        self.mesh = Mesh(mesh_file)

    def Space(self):
        self.fes = Periodic(VectorH1(self.mesh, order=self.order, dirichlet="bot"))
        print(f'sum(fes.FreeDofs()) = {sum(self.fes.FreeDofs())}')

    def model(self):
        u = self.fes.TrialFunction()
        I = Id(self.mesh.dim)
        F = I + Grad(u)
        
        def negpart(var):
            return (sqrt(var**2)-var)*0.5        

        AA = 1e5
        self.a = BilinearForm(self.fes, symmetric=False)
        self.a += Variation(self.gel.W(F).Compile() * dx)
        self.a += Variation(AA*negpart(y+u[1])**2 * dx)