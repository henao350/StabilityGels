import datetime
from ngsolve import *
import numpy as np
import scipy.optimize
import sys

class gel_3D:
    def __init__(self, folder_name_suffix, length=90.0, thickness=3.0, width =23.5):
        self.phi0 = 0.2
        self.entropic_unit = 136.6  # measured in MPa
        self.G = 0.13               # measured in MPa
        self.gamma = self.G/self.entropic_unit # 0.0009516837481698391 self.compute_gamma( lamb =1.4874):
        self.lambda_target = 1.99  # 0.13 = Gfun(1.99)
        self.chi =  0.348 # compute_chi(phi0=0.2035, gamma =0.0009516837481698391, J=J_iso)
        self.density =  1.23 # measured in [g/mL]

        self.L = length      # measured in mm
        self.d = thickness    # measured in mm
        self.w = width # measured in mm
        # self.delta = delta   # dimensionless
        
        # self.filename_suffix = "_d={:.2f}_delta={:.3f}".format(self.d, self.delta)
        self.filename_suffix = "_d={:.2f}".format(self.d)
        self.folder_name_suffix = folder_name_suffix; 

        def auxIsotropic(s):
            return s*self.dH(s*s*s) + self.gamma
        self.lambda_iso = scipy.optimize.fsolve(auxIsotropic, 1.7)
        
        def auxEnergyDensity(lambda1, lambda2, lambda3):
            gel=self; phi0=gel.phi0; G=gel.G; chi=gel.chi; nu=gel.entropic_unit
            J= lambda1*lambda2*lambda3;
            phi = phi0/J;
            return 0.5*G*(lambda1**2 + lambda2**2 + lambda3**2) + nu*((J-phi0)*np.log(1-phi) + phi0*chi*(1-phi))

        lambda_iso = self.lambda_iso
        self.reference_energy_density = auxEnergyDensity(lambda_iso, lambda_iso, lambda_iso)[0]

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
        # Finite element space with zero-displacement boundary condition on the bonded
        # part of the gel/substrate interface, and zero-vertical-displacement boundary condition
        # on the debonded part.
        # self.fes = VectorH1(self.mesh, order=self.order, dirichlet="bonded", dirichlety="debonded")
        self.fes = VectorH1(self.mesh, order=self.order, dirichlet="bonded|debonded")
        print('nDoF = {}'.format(self.fes.ndof))
        
    def model(self):
        u  = self.fes.TrialFunction()
        I = Id(self.mesh.dim)
        F = I + Grad(u)
        # Ft = I+ Grad(u).Trace()

        gravity = CoefficientFunction((0,-9.8,0))
        gel_density_in_g_per_mL = self.gel.density

        def negpart(var):
            #return max(-var,0)
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
        
        G_end = self.gel.G
        G0 = self.gel.Gfun(1.1);
        nIterations = 15
        self.gel.G = Parameter(G0)
        self.model()
        
        #From 0 to nIterations-1
        lambda_list = np.linspace(1.1, self.gel.lambda_target, nIterations, endpoint = False)
        G_list = [self.gel.Gfun(la) for la in lambda_list]
        
        # final iteration
        G_list.append(G_end)
              
        filename='gridfunctions' + self.gel.folder_name_suffix + \
            '/result_debonded3D'+self.gel.filename_suffix+ \
            "_order={}".format(self.order)

        tol=1e-3; maxits=100;
        
        indexes_iterations = range(nIterations+1)
        for numIteration in indexes_iterations:
            G_i = G_list [numIteration]
            print("*** Iteration #", numIteration, ". Shear modulus G = ", G_i)

            if numIteration==nIterations:
                tol=1e-6; maxits=500;
            
            self.gel.G.Set(G_i)
            self.gfu, _,_ = SolveNonlinearMinProblem_experimental(a= self.a, gfu = self.gfu,\
                        FreeDofs =self.fes.FreeDofs(), maxits=maxits, tol=tol, alpha=3e-2)

            # self.gfu.Save(filename + '_iter=' + str(numIteration).zfill(2) + '.gfu')
            self.gfu.Save(filename + '.gfu')
            print("Total time elapsed =" + str(datetime.datetime.now() - self.start_time))

# experimental acceleration
def SolveNonlinearMinProblem_experimental(a, gfu, FreeDofs, tol=1e-08, maxits=50, alpha=5e-2):#, scenes=None):
    
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

# Solves for the COMPLETELY BONDED gel (no delta)

# Most commonly changed parameters

# L= 90.0
# d = 3.00
# w = 23.5
# order= 2                # polynomial degree of finite elements

# Ex. if d=3.00 then the list of deltas where the indexes_deltas 
#   are given the corresponding indexes for the simularion
#   is in the file 'meshes3_00/deltas'.
# For example, the delta with index 93 for d=3.00 is delta=0.913

# It will be assumed that the arguments passed from shell
# are, in that order:
# the length, the thickness, and the order for the finite element space
# For example,
#    90 3.00 23.5 2 1
# means L=90 mm, d=3.00 mm, w=23.5 mm, order=2, coarse_flag=1
# If coarse_flag is different from 0, then it seeks the mesh in the folder 
# coarsemeshes1_62/3_00 and saves the results in coarsegridfunctions1_62/3_00

# data = [dummy, L, d, w, order, coarse_flag]
# data = ['','90','3.00','23.5', '2', '1'] 

data = sys.argv

L = float(data[1])
d = float(data[2])
w = float(data[3])
order = int(data[4])
coarse_flag = int(data[5])
# index_delta = int(data[6])

print('L={}, d={}, w={}, order={}, coarse_flag={}'.format(L, d,w, order,coarse_flag))

### Loads file 'deltas', containing the values of delta
folder_name_suffix = str(int(d)) + '_' + str(int(d%1*100)).zfill(2)
if coarse_flag!=0:
    folder_name_suffix += '_coarse'
filename = 'meshes' + folder_name_suffix + '/deltas'
delta_values = np.loadtxt(filename)

# indexes_deltas = [0,1,2]  
# for index_delta in indexes_deltas:    
# delta = delta_values[index_delta];
# print("Mesh number = {}, delta={:.3f}".format(index_delta,delta))
gel = gel_3D(folder_name_suffix = folder_name_suffix, length=L, thickness=d,width=w)
modelling = Solve_gel3d (gel, order=order)
###
index_delta=0   # it can be any index between 0 and 9
mesh_file = 'meshes' + gel.folder_name_suffix + '/mesh{}.vol.gz'.format(index_delta);
modelling.add_mesh(mesh_file)
###
modelling.Solve_incremental_softening()
