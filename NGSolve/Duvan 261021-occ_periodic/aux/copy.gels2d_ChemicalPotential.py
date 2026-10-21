from ngsolve import *
import numpy as np
import scipy.optimize
import datetime

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
        print(f'gel.Gfun(lambda_target) = {self.Gfun(self.lambda_target)}')

        self.filename_suffix = f"_phi0={phi0}_absMuBar={abs(mu_bar):.3f}"
        print(f'filename_suffix='+ self.filename_suffix)

    def phi(self, J):
        return self.phi0/J

    def H(self, J): # Flory-Huggins energy density in Lagrangian coordinates (energy per reference volume)
        return (J - self.phi0)*log(1-self.phi(J))  + self.phi0 * self.chi*(1-self.phi(J))

    def dH(self, J):
        return self.phi(J) + np.log(1-self.phi(J)) + self.chi * self.phi(J)**2

    # Computes gamma (i.e. G/entropic_unit) from a desired target value lamb for lambda_uniaxial
    # and a given mu_bar and p_bar, using [Kang & Huang JMPS 2010, Eq.(3.6)]
    def Gfun(self, lamb):
        mu_bar = self.mu_bar.Get()
        p_bar = self.p0_bar*np.exp(mu_bar)
        return ((mu_bar - p_bar -self.dH(lamb))/(lamb-1/lamb))*self.entropic_unit

    def find_lambda_target(self):
        G_value = self.G.Get()
        def auxFunctionLambdaTargetShearModulus(lamb):
            return self.Gfun(lamb) - G_value
        # Among 100 trials, chooses the value of lambda
        # for which Gfun is as the closest to the target G_value
        lambda_start = 1.0001; lambda_end = 10; numPoints=1000;
        lambda_values = np.linspace(lambda_start, lambda_end, numPoints)        
        lambda_initial_fsolve = lambda_start
        best_abs_dist= abs(auxFunctionLambdaTargetShearModulus(lambda_start))
        for lamb in lambda_values:
            dist = auxFunctionLambdaTargetShearModulus(lamb)
            if abs(dist) < best_abs_dist:
                best_abs_dist = abs(dist)
                lambda_initial_fsolve = lamb   
        # and finally, for higher precision, fsolve is called
        # This seems to be necessary since the functin Gfun is
        # extremely sensitive to changes in lambda even in its fourth decimal place
        lambda_target = (scipy.optimize.fsolve(auxFunctionLambdaTargetShearModulus, lambda_initial_fsolve))[0]   # approximate solution to  self.G_fun(lamb) = G_value, for given G_value          
        return lambda_target

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
        # hydrogel model        
        AA = 1e5
        self.a = BilinearForm(self.fes, symmetric=False)
        self.a += Variation(  self.gel.W(F).Compile() * dx)
        self.a += Variation(  AA*negpart(y+u[1])**2 * dx)

    def Solve_incremental_softening(self):
        self.Space()        
        self.model()
        #
        self.gfu = GridFunction(self.fes)
        self.gfu.vec[:] = 0
        #
        G_end = self.gel.G.Get()
        G0 = self.gel.Gfun(1.1)
        self.gel.G.Set(G0)
        print(f'G0={self.gel.G.Get()}')
        lambda_initial = 1.1
        nIterations = 15
        #From 0 to nIterations-1
        lambda_list = np.linspace(lambda_initial, self.gel.lambda_target, nIterations+1, endpoint = True)
        G_list = [self.gel.Gfun(la) for la in lambda_list[:-1]]
        # final iteration
        G_list.append(G_end)

        filename = 'gridfunctions/result_gelsMu2D'+self.gel.filename_suffix + f'_order={self.order}'
        print(f'filename={filename}')    

        tol=1e-3; maxits=100;
        indexes_iterations = range(nIterations+1)
        for numIteration in indexes_iterations:
            G_i = G_list[numIteration]            
            print(f"*** Iteration #{numIteration}, G = {G_i}, lambda_uniaxial={lambda_list[numIteration]}")
            if numIteration==nIterations:
                tol=1e-6; maxits=500;   
            self.gel.G.Set(G_i)            
            self.gfu, _,_ = SolveNonlinearMinProblem(a= self.a, gfu = self.gfu,\
                        FreeDofs =self.fes.FreeDofs(), maxits=maxits, tol=tol)
            F=Id(2)+Grad(self.gfu)
            print('Average energy density = ', round(Integrate(self.gel.W(F), self.mesh)/(L*d),5))
            vertex_values = [(y+self.gfu[1])(self.mesh(*p)) for p in self.mesh.ngmesh.Points()]
            print(f'Maximum vertical stretch: ', max(vertex_values)/self.gel.d)
            # Draw(self.gfu, deformation=True)            
            # self.gfu.Save(filename + '_iter=' + str(numIteration).zfill(2) + '.gfu')
            self.gfu.Save(filename + '.gfu')


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
        #alpha = 5e-2
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
    print(" ")
    # 

