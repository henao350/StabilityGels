from ngsolve import *
import numpy as np
import scipy.optimize
import datetime
import sys


# =========================================================
# GEL MODEL
# =========================================================

class gel_3D:

    def __init__(
        self,
        length=90.0,
        width=15.0,
        thickness=1.6,
        phi0=0.2,
        mu_bar=-0.05
    ):

        self.phi0 = phi0

        # -------------------------------------------------
        # thermodynamics
        # -------------------------------------------------

        T = 25 + 273.15
        K_B = 1.380649e-23
        V_m = 3e-29

        self.entropic_unit = K_B*T/V_m*1e-6

        self.gamma = 0.001
        self.chi = 0.4

        self.G = self.gamma*self.entropic_unit

        vapor_pressure = 3.2e-3

        self.p0_bar = vapor_pressure/self.entropic_unit

        self.mu_bar = Parameter(mu_bar)

        self.p_bar = Parameter(
            self.p0_bar*np.exp(mu_bar)
        )

        # -------------------------------------------------

        self.L = length
        self.w = width
        self.d = thickness

        print(f'phi0 = {phi0}')
        print(f'mu_bar = {mu_bar}')

    # =====================================================
    # NUMPY FUNCTIONS
    # =====================================================

    def phi_numpy(self, J):
        return self.phi0/J

    def dH_numpy(self, J):

        if J <= self.phi0:
            return 1e20

        phi = self.phi_numpy(J)

        mu = float(self.mu_bar.Get())
        p = float(self.p_bar.Get())

        return (
            phi
            + np.log(1-phi)
            + self.chi*phi**2
            - self.gamma/J
            + p
            - mu
        )

    # =====================================================
    # NGSOLVE FUNCTIONS
    # =====================================================

    def phi(self, J):
        return self.phi0/J

    def H(self, J):

        phi = self.phi(J)

        eps = 1e-12

        return (
            (J-self.phi0)*log(Max(1-phi, eps))
            + self.phi0*self.chi*(1-phi)
            - self.gamma*log(Max(J, eps))
            + (self.p_bar-self.mu_bar)*(J-self.phi0)
        )

    def W(self, F):

        J = Det(F)

        C = F.trans * F

        return (
            0.5*self.G*(Trace(C)-3)
            + self.entropic_unit*self.H(J)
        )

    # =====================================================
    # CHEMICAL POTENTIAL
    # =====================================================

    def mu_fun(self, lamb):

        phi0 = self.phi0
        gamma = self.gamma
        chi = self.chi
        p0_bar = self.p0_bar

        def residual(mu_bar):

            p_bar = p0_bar*np.exp(mu_bar)

            phi = phi0/lamb

            if phi >= 1:
                return 1e20

            return (
                phi
                + np.log(1-phi)
                + chi*phi**2
                - gamma/lamb
                + p_bar
                - mu_bar
            )

        mu_guess = -0.05

        mu_sol = scipy.optimize.fsolve(
            residual,
            mu_guess
        )[0]

        return mu_sol


# =========================================================
# SOLVER
# =========================================================

class Solve_gel3d:

    def __init__(self, gel, order=2):

        self.gel = gel
        self.order = order

        self.start_time = datetime.datetime.now()

    def add_mesh(self, mesh_file):

        self.mesh = Mesh(mesh_file)

    def Space(self):

        self.fes = VectorH1(
            self.mesh,
            order=self.order,
            dirichlet="bonded|debonded"
        )

        print('nDoF =', self.fes.ndof)

    def model(self):

        u = self.fes.TrialFunction()

        I = Id(self.mesh.dim)

        F = I + Grad(u)

        def negpart(var):
            return 0.5*(sqrt(var**2)-var)

        AA = 1e5

        self.a = BilinearForm(
            self.fes,
            symmetric=False
        )

        self.a += Variation(
            self.gel.W(F)*dx
        )

        self.a += Variation(
            AA*negpart(y + u[1])**2 * dx
        )


# =========================================================
# NEWTON SOLVER
# =========================================================

def SolveNonlinearMinProblem(
    a,
    gfu,
    tol=1e-8,
    maxits=50,
    alpha=1e-2
):

    start_time = datetime.datetime.now()

    res = gfu.vec.CreateVector()
    du = gfu.vec.CreateVector()

    precond = 'bddc'

    c = Preconditioner(a, precond)

    for it in range(maxits):

        with TaskManager():

            a.Apply(gfu.vec, res)

            a.AssembleLinearization(gfu.vec)

            c.Update()

            inv = CGSolver(
                a.mat,
                c.mat,
                maxsteps=1000
            )

            du.data = alpha * inv * res

            gfu.vec.data -= du

        stopcritval = sqrt(
            abs(InnerProduct(du, res))
        )

        print(
            "Newton iteration:",
            it,
            "Residual:",
            stopcritval
        )

        if stopcritval < tol:
            break

    return gfu, stopcritval, it


# =========================================================
# MAIN
# =========================================================

data = sys.argv

if len(data) < 6:

    data = [
        '',
        '90',
        '15.0',
        '1.62',
        '0.2',
        '0.05'
    ]

order = 2

mesh_file = 'meshes/mesh0.vol.gz'

L = float(data[1])
w = float(data[2])
d = float(data[3])
phi0 = float(data[4])

mu_input = float(data[5])

mu_bar = -mu_input

print(f'L={L}')
print(f'w={w}')
print(f'd={d}')
print(f'phi0={phi0}')
print(f'mu_bar={mu_bar}')

gel = gel_3D(
    length=L,
    width=w,
    thickness=d,
    phi0=phi0,
    mu_bar=mu_bar
)

modelling = Solve_gel3d(
    gel,
    order=order
)

modelling.add_mesh(mesh_file)

# =========================================================
# MANUAL MODE
# =========================================================

la = 1.1

mu_bar = modelling.gel.mu_fun(la)

print(f'computed mu_bar = {mu_bar}')

modelling.Space()

modelling.gfu = GridFunction(
    modelling.fes
)

u0 = CoefficientFunction(
    (0, (la-1.0)*y, 0)
)

modelling.gfu.Set(u0)

modelling.gel.mu_bar.Set(mu_bar)

modelling.gel.p_bar.Set(
    modelling.gel.p0_bar*np.exp(mu_bar)
)

modelling.model()

tol = 1e-3
maxits = 100

modelling.gfu, _, _ = SolveNonlinearMinProblem(
    a=modelling.a,
    gfu=modelling.gfu,
    maxits=maxits,
    tol=tol,
    alpha=1e-2
)

filename_suffix = (
    f'_phi0={phi0:.1f}'
    f'_muBarAbs={np.abs(mu_bar):.6f}'
)

filename = (
    'gridfunctions/result'
    + filename_suffix
    + f'_order={order}'
)

filename += f'_manualMode_la={la}'

modelling.gfu.Save(
    filename + '.gfu'
)

print("Simulation finished.")