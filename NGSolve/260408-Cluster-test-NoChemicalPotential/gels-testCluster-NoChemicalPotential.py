from ngsolve import *
# from ngsolve.webgui import Draw
import numpy as np
import scipy.optimize
import datetime
import sys

class gel_3D:
    def __init__(self, length=90.0, width =15.0, thickness=1.6, phi0=0.2):
        self.phi0 = phi0
        self.entropic_unit = 136.6  # measured in MPa
        self.G = 0.13               # measured in MPa
        self.gamma = self.G/self.entropic_unit
        self.chi =  0.348
        self.density =  1.23 # measured in [g/mL]

        self.L = length      # measured in mm
        self.d = thickness    # measured in mm
        self.w = width # measured in mm
        # self.delta = delta   # dimensionless

        self.filename_suffix = "_phi0={:.1f}_mesh0Polymer".format(self.phi0)

        def auxIsotropic(s):
            return s*self.dH(s*s*s) + self.gamma
        self.lambda_iso = scipy.optimize.fsolve(auxIsotropic, 1.7)[0]

        def auxUniaxial(s):
            return s*self.gamma + self.dH(s)
        self.lambda_target = scipy.optimize.fsolve(auxUniaxial, 1.9)[0]
        
        def auxEnergyDensity(lambda1, lambda2, lambda3):
            gel=self; phi0=gel.phi0; G=gel.G; chi=gel.chi; nu=gel.entropic_unit
            J= lambda1*lambda2*lambda3;
            phi = phi0/J;
            return 0.5*G*(lambda1**2 + lambda2**2 + lambda3**2) + nu*((J-phi0)*np.log(1-phi) + phi0*chi*(1-phi))

        lambda_iso = self.lambda_iso
        self.reference_energy_density = auxEnergyDensity(lambda_iso, lambda_iso, lambda_iso)                

    def phi(self, J):
        return self.phi0/J

    def H(self, J):
        return (J - self.phi0)*log(1-self.phi(J))  + self.phi0 * self.chi*(1-self.phi(J))

    def dH(self, J):
        return self.phi(J) + np.log(1-self.phi(J)) + self.chi * self.phi(J)**2

    def Gfun(self, lamb):
        nu = self.entropic_unit
        return (-self.dH(lamb)/lamb)*nu
        
    # energy density in [MPa]
    def W(self, F):
               
        J = Det(F)
        C = F.trans* F
        
        gel = self
        G = gel.G
        nu = gel.entropic_unit
        
        reference_energy_density =  self.reference_energy_density
        
        return 0.5*G*(Trace(C)) + nu*gel.H(J) - reference_energy_density


class Solve_gel3d:
    def __init__(self, gel, order=2):         # corner_refinement capability not yet activated
        self.gel = gel
        self.order = order
        self.start_time = datetime.datetime.now()  

    def add_mesh(self, mesh_file):        
        self.mesh = Mesh(mesh_file)
    
    def Space(self):
        # Finite element space with zero-displacement boundary condition 
        # on all of the bottom interface
        self.fes = VectorH1(self.mesh, order=self.order, dirichlet="bonded|debonded")
        print('nDoF = {}'.format(self.fes.ndof))
        
    def model(self):
        u  = self.fes.TrialFunction()
        I = Id(self.mesh.dim)
        F = I + Grad(u)

        gravity = CoefficientFunction((0,0,-9.8))
        gel_density_in_g_per_mL = self.gel.density

        def negpart(var):
            #return max(-var,0)Nonlinea
            return (sqrt(var**2)-var)*0.5        
        
        # Verificar este parametro
        AA = 1e5

        # hydrogel model        
        self.a = BilinearForm(self.fes, symmetric=False)
        self.a += Variation(  self.gel.W(F).Compile() * dx)
        #verificar parametro 1e-6 parece ser el mismo que en 2d
        self.a += Variation( -((1e-6)*gel_density_in_g_per_mL)*InnerProduct(gravity, u)*dx )   #in [mJ] the gravitational energy
        self.a += Variation(  AA*negpart(y+u[1])**2 * dx)
        
    def Solve_incremental_softening(self):
        self.Space()
        self.gfu = GridFunction(self.fes)
        self.gfu.vec[:] = 0
        
        lambda_initial = 1.1
        G_end = self.gel.G
        G0 = self.gel.Gfun(lambda_initial);
        nIterations = 15
        self.gel.G = Parameter(G0)
        self.model()
        
        #From 0 to nIterations-1
        lambda_list = np.linspace(lambda_initial, self.gel.lambda_target, nIterations, endpoint = False)
        G_list = [self.gel.Gfun(la) for la in lambda_list]
        
        # final iteration
        G_list.append(G_end)
              
        filename='gridfunctions/result'+ \
            self.gel.filename_suffix+ \
            "_order={}".format(self.order)

        tol=1e-3; maxits=100;
        
        indexes_iterations = range(nIterations+1)
        #indexes_iterations = [0]
        for numIteration in indexes_iterations:
            G_i = G_list [numIteration]
            print("*** Iteration #", numIteration, ". Shear modulus G = ", G_i)

            if numIteration==nIterations:
                tol=1e-6; maxits=500;
            
            self.gel.G.Set(G_i)
            self.gfu, _,_ = SolveNonlinearMinProblem(a= self.a, gfu = self.gfu, maxits=maxits, tol=tol, alpha=1e-2)

            self.gfu.Save(filename + '_iter=' + str(numIteration).zfill(2) + '.gfu')
            # self.gfu.Save(filename + '.gfu')
            print("Total time elapsed =" + str(datetime.datetime.now() - self.start_time))

# experimental acceleration
def SolveNonlinearMinProblem(a, gfu, tol=1e-08, maxits=50, alpha=5e-2):
    
    start_time = datetime.datetime.now()  

    res = gfu.vec.CreateVector()
    du  = gfu.vec.CreateVector()
    
    # precond local, multigrid, bddc
    precond = 'bddc'
    c = Preconditioner(a, precond)
    for it in range(maxits):
        #print ("Newton iteration {:3}".format(it),end=", ")
        # print ("energy = {:16}".format(a.Energy(gfu.vec)),end="")

        # solve linearized problem:

        with TaskManager():
            a.Apply (gfu.vec, res)
            c = Preconditioner(a, precond)
            a.AssembleLinearization (gfu.vec)
            
            # instead of computing inverse try CG solver with preconditioner
            #inv = a.mat.Inverse(FreeDofs)
            c.Update()
            inv = CGSolver(a.mat, c.mat, maxsteps=1000)
            # try adaptive alpha
            du.data = alpha * inv * res

            #update iteration
            gfu.vec.data -= du

        #stopping criteria
        stopcritval = sqrt(abs(InnerProduct(du,res)))
        print("Newton iteration:", it, "Time elapsed =" + str(datetime.datetime.now() - start_time), end= "\n")
        print ("<A u",it,", A u",it,">_{-1}^0.5 = ", stopcritval)
        if stopcritval < tol:
            
            break

    return gfu, stopcritval, it


### MAIN ###

# data = [dummy, L, w, d, phi0]
# data = ['','90', '15.0', '1.62', '0.2'] 
data = sys.argv

order = 2
# order = 1
# maxh = 0.41
mesh_file = 'meshes/mesh0.vol.gz'

L = float(data[1])
d = float(data[2])
w = float(data[3])
phi0 = float(data[4])

print(f'L={L}, w={w}, d={d}, phi0={phi0}')

gel = gel_3D(length=L, width=w, thickness=d, phi0=phi0)
modelling = Solve_gel3d (gel, order=order)
modelling.add_mesh(mesh_file)
modelling.Space()
modelling = Solve_gel3d (gel, order=order)
modelling.add_mesh(mesh_file)
modelling.Solve_incremental_softening()
