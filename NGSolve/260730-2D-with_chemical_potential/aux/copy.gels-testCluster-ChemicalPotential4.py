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
        mu_bar=-0.03
    ):

        self.phi0 = phi0

        # -------------------------------------------------
        # thermodynamic constants
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

        # use NGSolve Parameters
        self.mu_bar = Parameter(mu_bar)

        self.p_bar = Parameter(
            self.p0_bar*np.exp(mu_bar)
        )

        # geometry
        self.L = length
        self.w = width
        self.d = thickness

        print(f'phi0 = {self.phi0}')
        print(f'mu_bar = {mu_bar}')
        print(f'entropic_unit = {self.entropic_unit}')
        print(f'gamma = {self.gamma}')
        print(f'chi = {self.chi}')

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

        # safe log(1-phi)

        one_minus_phi_safe = IfPos(
            1 - phi - eps,
            1 - phi,
            eps
        )

        # safe log(J)

        J_safe = IfPos(
            J - eps,
            J,
            eps
        )

        return (
            (J-self.phi0)*log(one_minus_phi_safe)
            + self.phi0*self.chi*(1-phi)
            - self.gamma*log(J_safe)
            + (self.p_bar-self.mu_bar)*(J-self.phi0)
        )

    # =====================================================
    # ENERGY
    # =====================================================

    def W(self, F):

        J = Det(F)

        C = F.trans * F

        return (
            0.5*self.G*(Trace(C)-3)
            + self.entropic_unit*self.H(J)
        )

    # =====================================================
    # COMPUTE MU FROM LAMBDA
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

        mu_guess = -0.03

        mu_sol = scipy.optimize.fsolve(
            residual,
            mu_guess
        )[0]

        print(f'lambda = {lamb}, computed mu = {mu_sol}')

        return mu_sol


# =========================================================
# SOLVER CLASS
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

        # hydrogel energy

        self.a += Variation(
            self.gel.W(F)*dx
        )

        # contact penalty

        self.a += Variation(
            AA*negpart(y+u[1])**2 * dx
        )


# =========================================================
# NONLINEAR SOLVER
# =========================================================

def SolveNonlinearMinProblem(
    a,
    gfu,
    tol=1e-8,
    maxits=50,
    alpha=1e-3
):

    start_time = datetime.datetime.now()

    res = gfu.vec.CreateVector()

    du = gfu.vec.CreateVector()

    w = gfu.vec.CreateVector()

    precond = 'bddc'

    c = Preconditioner(a, precond)

    for it in range(maxits):

        with TaskManager():

            # residual

            a.Apply(gfu.vec, res)

            # Jacobian

            a.AssembleLinearization(gfu.vec)

            c.Update()

            inv = CGSolver(
                a.mat,
                c.mat,
                maxsteps=1000
            )

            du.data = inv * res

        # =================================================
        # BACKTRACKING LINE SEARCH
        # =================================================

        step = alpha

        success = False

        res_norm_old = sqrt(
            abs(InnerProduct(res, res))
        )

        for ls in range(12):

            w.data = gfu.vec - step*du

            with TaskManager():

                a.Apply(w, res)

                res_norm_new = sqrt(
                    abs(InnerProduct(res, res))
                )

            if np.isnan(res_norm_new):

                step *= 0.5
                continue

            if res_norm_new < res_norm_old:

                success = True
                break

            step *= 0.5

        if not success:

            print("WARNING: line search failed")

        # update

        gfu.vec.data = w

        stopcritval = sqrt(
            abs(InnerProduct(du, res))
        )

        print("")
        print("Newton iteration:", it)
        print("Residual =", res_norm_new)
        print("Step =", step)
        print("Elapsed =", datetime.datetime.now()-start_time)

        if stopcritval < tol:

            print("Newton converged")

            break

    return gfu, stopcritval, it


# =========================================================
# MAIN
# =========================================================

data = sys.argv

if len(data) < 6:

    print("Using default parameters")

    data = [
        '',
        '90',
        '15.0',
        '1.62',
        '0.20',
        '0.03'
    ]

# =========================================================
# INPUT PARAMETERS
# =========================================================

order = 2

mesh_file = 'meshes/mesh0.vol.gz'

L = float(data[1])

w = float(data[2])

d = float(data[3])

phi0 = float(data[4])

# positive in txt file
mu_input = float(data[5])

# internal sign convention
mu_bar = -mu_input

print(f'L={L}')
print(f'w={w}')
print(f'd={d}')
print(f'phi0={phi0}')
print(f'mu_bar={mu_bar}')

# =========================================================
# CREATE GEL
# =========================================================

gel = gel_3D(
    length=L,
    width=w,
    thickness=d,
    phi0=phi0,
    mu_bar=mu_bar
)

# =========================================================
# CREATE MODEL
# =========================================================

modelling = Solve_gel3d(
    gel,
    order=order
)

modelling.add_mesh(mesh_file)

modelling.Space()

modelling.gfu = GridFunction(
    modelling.fes
)

# =========================================================
# INITIAL CONDITION
# =========================================================

la = 1.05

if modelling.mesh.dim == 3:

    u0 = CoefficientFunction(
        (0, (la-1.0)*y, 0)
    )

else:

    u0 = CoefficientFunction(
        (0, (la-1.0)*y)
    )

modelling.gfu.Set(u0)

# =========================================================
# BUILD MODEL
# =========================================================

modelling.model()

# =========================================================
# CONTINUATION IN MU
# =========================================================

nIterations = 15

mu_start = -0.15

mu_end = mu_bar

mu_list = np.linspace(
    mu_start,
    mu_end,
    nIterations
)

filename_suffix = (
    f'_phi0={phi0:.2f}'
    f'_muBarAbs={np.abs(mu_bar):.6f}'
)

filename = (
    'gridfunctions/result'
    + filename_suffix
    + f'_order={order}'
)

# =========================================================
# ITERATIONS
# =========================================================

tol = 1e-3

maxits = 100

for numIteration in range(nIterations):

    mu_i = mu_list[numIteration]

    print("")
    print("====================================")
    print("Continuation iteration =", numIteration)
    print("mu =", mu_i)
    print("====================================")

    # update parameters

    modelling.gel.mu_bar.Set(mu_i)

    modelling.gel.p_bar.Set(
        modelling.gel.p0_bar*np.exp(mu_i)
    )

    # tighter final solve

    if numIteration == nIterations-1:

        tol = 1e-6

        maxits = 500

    # solve nonlinear problem

    modelling.gfu, _, _ = SolveNonlinearMinProblem(
        a=modelling.a,
        gfu=modelling.gfu,
        maxits=maxits,
        tol=tol,
        alpha=1e-3
    )

    # save intermediate result

    filename_iter = (
        filename
        + "_iter="
        + str(numIteration).zfill(2)
    )

    modelling.gfu.Save(
        filename_iter + ".gfu"
    )

    print("")
    print("Saved:")
    print(filename_iter + ".gfu")

# =========================================================
# END
# =========================================================

print("")
print("Simulation finished.")