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

        # thermodynamic constants
        T = 25 + 273.15
        K_B = 1.380649e-23
        V_m = 3e-29

        self.entropic_unit = K_B * T / V_m * 1e-6

        self.gamma = 0.001
        self.chi = 0.4

        # IMPORTANT:
        # use Parameter for continuation
        self.mu_bar = Parameter(mu_bar)

        vapor_pressure = 3.2e-3
        self.p0_bar = vapor_pressure / self.entropic_unit

        self.p_bar = Parameter(
            self.p0_bar * np.exp(mu_bar)
        )

        self.G = 0.001 * self.entropic_unit

        self.L = length
        self.w = width
        self.d = thickness

        self.filename_suffix = (
            f"_phi0={phi0:.2f}"
            f"_muBarAbs={abs(mu_bar):.4f}"
        )

        # -------------------------------------------------
        # isotropic equilibrium
        # -------------------------------------------------

        def aux_iso(s):
            return s * self.dH_numpy(s**3) + self.gamma

        self.lambda_iso = scipy.optimize.fsolve(aux_iso, 1.2)[0]

        # -------------------------------------------------
        # target uniaxial equilibrium
        # -------------------------------------------------

        def aux_uni(s):
            return s * self.gamma + self.dH_numpy(s)

        self.lambda_target = scipy.optimize.fsolve(aux_uni, 1.2)[0]

        print("lambda_iso =", self.lambda_iso)
        print("lambda_target =", self.lambda_target)

        # reference energy

        self.reference_energy_density = self.energy_density_numpy(
            self.lambda_iso,
            self.lambda_iso,
            self.lambda_iso
        )

    # =====================================================
    # NUMPY FUNCTIONS
    # =====================================================

    def phi_numpy(self, J):
        return self.phi0 / J

    def dH_numpy(self, J):

        phi = self.phi_numpy(J)

        mu = float(self.mu_bar.Get())
        p = float(self.p_bar.Get())

        return (
            phi
            + np.log(1 - phi)
            + self.chi * phi**2
            - self.gamma / J
            + p
            - mu
        )

    def energy_density_numpy(self, l1, l2, l3):

        J = l1 * l2 * l3

        phi = self.phi_numpy(J)

        mu = float(self.mu_bar.Get())
        p = float(self.p_bar.Get())

        return (
            0.5 * self.G * (l1**2 + l2**2 + l3**2 - 3)
            + self.entropic_unit
            * (
                (J - self.phi0) * np.log(1 - phi)
                + self.phi0 * self.chi * (1 - phi)
                - self.gamma * np.log(J)
                + (p - mu) * (J - self.phi0)
            )
        )

    # =====================================================
    # NGSOLVE FUNCTIONS
    # =====================================================

    def phi(self, J):
        return self.phi0 / J

    def H(self, J):

        phi = self.phi(J)

        return (
            (J - self.phi0) * log(1 - phi)
            + self.phi0 * self.chi * (1 - phi)
            - self.gamma * log(J)
            + (self.p_bar - self.mu_bar) * (J - self.phi0)
        )

    def W(self, F):

        J = Det(F)
        C = F.trans * F

        return (
            0.5 * self.G * (Trace(C) - 3)
            + self.entropic_unit * self.H(J)
            - self.reference_energy_density
        )

    # =====================================================
    # continuation in mu
    # =====================================================

    def mu_fun(self, lamb):

        phi0 = self.phi0
        gamma = self.gamma
        chi = self.chi
        p0_bar = self.p0_bar

        def residual(mu_bar):

            p_bar = p0_bar * np.exp(mu_bar)

            phi = phi0 / lamb

            return (
                phi
                + np.log(1 - phi)
                + chi * phi**2
                - gamma / lamb
                + p_bar
                - mu_bar
            )

        mu_guess = -0.05

        return scipy.optimize.fsolve(residual, mu_guess)[0]


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

        print("nDoF =", self.fes.ndof)

    def model(self):

        u = self.fes.TrialFunction()

        I = Id(self.mesh.dim)

        F = I + Grad(u)

        def negpart(var):
            return 0.5 * (sqrt(var**2) - var)

        AA = 1e5

        self.a = BilinearForm(
            self.fes,
            symmetric=False
        )

        self.a += Variation(
            self.gel.W(F) * dx
        )

        self.a += Variation(
            AA * negpart(y + u[1])**2 * dx
        )

    def Solve_incremental_softening(self):

        self.Space()

        self.gfu = GridFunction(self.fes)

        lambda_initial = 1.02

        if self.mesh.dim == 3:
            u0 = CoefficientFunction(
                (0, (lambda_initial - 1.0) * y, 0)
            )
        else:
            u0 = CoefficientFunction(
                (0, (lambda_initial - 1.0) * y)
            )

        self.gfu.Set(u0)

        nIterations = 15

        lambda_list = np.linspace(
            lambda_initial,
            self.gel.lambda_target,
            nIterations
        )

        mu_list = [
            self.gel.mu_fun(la)
            for la in lambda_list
        ]

        self.model()

        filename = (
            "gridfunctions/result"
            + self.gel.filename_suffix
            + f"_order={self.order}"
        )

        tol = 1e-3
        maxits = 100

        for k in range(nIterations):

            mu_i = mu_list[k]

            print("\n*** Iteration", k)
            print("mu_bar =", mu_i)

            if k == nIterations - 1:
                tol = 1e-6
                maxits = 500

            # update parameters
            self.gel.mu_bar.Set(mu_i)

            self.gel.p_bar.Set(
                self.gel.p0_bar * np.exp(mu_i)
            )

            self.gfu, _, _ = SolveNonlinearMinProblem(
                a=self.a,
                gfu=self.gfu,
                tol=tol,
                maxits=maxits,
                alpha=1e-2
            )

            self.gfu.Save(
                filename
                + "_iter="
                + str(k).zfill(2)
                + ".gfu"
            )

            print(
                "Total elapsed =",
                datetime.datetime.now()
                - self.start_time
            )


# =========================================================
# NONLINEAR SOLVER
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

    precond = "bddc"

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
            "residual =",
            stopcritval
        )

        if stopcritval < tol:
            break

    return gfu, stopcritval, it


# =========================================================
# MAIN
# =========================================================

data = sys.argv

order = 2

mesh_file = "meshes/mesh0.vol.gz"

L = float(data[1])
w = float(data[2])
d = float(data[3])
phi0 = float(data[4])

mu_bar = -float(data[5])

print(
    f"L={L}, w={w}, d={d}, "
    f"phi0={phi0}, mu_bar={mu_bar}"
)

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

modelling.Solve_incremental_softening()