import netgen.geom2d as geom2d
from ngsolve import *
from netgen.meshing import MeshingParameters, meshsize
from ngsolve.webgui import Draw

import netgen.meshing as ngmsh

import numpy as np
import scipy.optimize
import csv
import datetime
import sys


# =========================================================
# 2D GEL WITH CHEMICAL POTENTIAL
# =========================================================

class gel_debonded2D:

    def __init__(
        self,
        folder_name_suffix,
        length=90.0,
        thickness=3.0,
        delta=0.9,
        phi0=0.2,
        mu_bar=-0.03
    ):

        self.phi0 = phi0

        # =====================================================
        # THERMODYNAMIC PARAMETERS
        # =====================================================

        T = 25 + 273.15
        K_B = 1.380649e-23
        V_m = 3e-29

        self.entropic_unit = K_B*T/V_m*1e-6

        self.gamma = 0.001

        self.chi = 0.4

        self.G = self.gamma*self.entropic_unit

        self.density = 1.23

        vapor_pressure = 3.2e-3

        self.p0_bar = vapor_pressure/self.entropic_unit

        self.mu_bar = Parameter(mu_bar)

        self.p_bar = Parameter(
            self.p0_bar*np.exp(mu_bar)
        )

        # =====================================================
        # GEOMETRY
        # =====================================================

        self.L = length

        self.d = thickness

        self.delta = delta

        self.folder_name_suffix = folder_name_suffix

        self.filename_suffix = (
            "_d={:.2f}_delta={:.3f}_muBarAbs={:.6f}"
        ).format(
            self.d,
            self.delta,
            np.abs(mu_bar)
        )

        print(f'phi0 = {self.phi0}')
        print(f'mu_bar = {mu_bar}')
        print(f'gamma = {self.gamma}')
        print(f'chi = {self.chi}')

    # =========================================================
    # VOLUME FRACTION
    # =========================================================

    def phi(self, J):

        return self.phi0/J

    # =========================================================
    # FREE ENERGY
    # =========================================================

    def H(self, J):

        phi = self.phi(J)

        eps = 1e-12

        one_minus_phi_safe = IfPos(
            1 - phi - eps,
            1 - phi,
            eps
        )

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

    # =========================================================
    # DERIVATIVE FOR CONTINUATION
    # =========================================================

    def dH_numpy(self, J):

        if J <= self.phi0:
            return 1e20

        phi = self.phi0/J

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

    # =========================================================
    # COMPUTE MU
    # =========================================================

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

        return mu_sol

    # =========================================================
    # ENERGY DENSITY
    # =========================================================

    def W(self, F):

        J = Det(F)

        C = F.trans * F

        return (
            0.5*self.G*(1.0 + Trace(C))
            + self.entropic_unit*self.H(J)
        )


# =========================================================
# SOLVER
# =========================================================

class Solve_gel_bonded:

    def __init__(self, gel, order=3):

        self.gel = gel

        self.order = order

    def add_mesh(self, index_delta):

        mesh_file = (
            'meshes'
            + self.gel.folder_name_suffix
            + '/mesh{}.vol'.format(index_delta)
        )

        self.mesh = Mesh(mesh_file)

    # =========================================================
    # FE SPACE
    # =========================================================

    def Space(self):

        self.fes = VectorH1(
            self.mesh,
            order=self.order,
            dirichlet="bonded_interface",
            dirichlety="debonded_interface"
        )

        print('nDoF = {}'.format(self.fes.ndof))

    # =========================================================
    # MODEL
    # =========================================================

    def model(self):

        u = self.fes.TrialFunction()

        I = Id(self.mesh.dim)

        F = I + Grad(u)

        gravity = CoefficientFunction((0,-9.8))

        gel_density_in_g_per_mL = self.gel.density

        def negpart(var):

            return (sqrt(var**2)-var)*0.5

        AA = 1e4

        self.a = BilinearForm(
            self.fes,
            symmetric=False
        )

        # hydrogel energy

        self.a += Variation(
            self.gel.W(F)*dx
        )

        # gravity

        self.a += Variation(
            -((1e-6)*gel_density_in_g_per_mL)
            *InnerProduct(gravity, u)*dx
        )

        # contact

        self.a += Variation(
            AA*negpart(y+u[1])**2 * dx
        )

    # =========================================================
    # CONTINUATION IN MU
    # =========================================================

    def Solve_incremental_softening(self):

        self.Space()

        self.gfu = GridFunction(self.fes)

        # =====================================================
        # INITIAL CONDITION
        # =====================================================

        la = 1.05

        amp = 0.001

        u0 = CoefficientFunction(
            (
                0,
                (la-1.0)*y
                + amp*sin(2*np.pi*x/self.gel.L)
            )
        )

        self.gfu.Set(u0)

        self.model()

        # =====================================================
        # CONTINUATION
        # =====================================================

        nIterations = 15

        mu_start = -0.15

        mu_end = float(self.gel.mu_bar.Get())

        mu_list = np.linspace(
            mu_start,
            mu_end,
            nIterations
        )

        filename = (
            'gridfunctions'
            + self.gel.folder_name_suffix
            + '/result_debonded2D'
            + self.gel.filename_suffix
            + "_order={}".format(self.order)
        )

        tol = 1e-3

        maxits = 100

        for numIteration in range(nIterations):

            mu_i = mu_list[numIteration]

            print("")
            print("================================")
            print("Iteration =", numIteration)
            print("mu =", mu_i)
            print("================================")

            # update chemical potential

            self.gel.mu_bar.Set(mu_i)

            self.gel.p_bar.Set(
                self.gel.p0_bar*np.exp(mu_i)
            )

            # final solve more accurate

            if numIteration == nIterations-1:

                tol = 1e-6

                maxits = 500

            # solve

            self.gfu, _, _ = SolveNonlinearMinProblem(
                a=self.a,
                gfu=self.gfu,
                FreeDofs=self.fes.FreeDofs(),
                maxits=maxits,
                tol=tol,
                alpha=1e-3
            )

            # save result

            self.gfu.Save(
                filename
                + '_iter='
                + str(numIteration).zfill(2)
                + '.gfu'
            )


# =========================================================
# NONLINEAR SOLVER
# =========================================================

def SolveNonlinearMinProblem(
    a,
    gfu,
    FreeDofs,
    tol=1e-8,
    maxits=50,
    alpha=1e-3
):

    res = gfu.vec.CreateVector()

    du = gfu.vec.CreateVector()

    w = gfu.vec.CreateVector()

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

            du.data = inv * res

        # =====================================================
        # LINE SEARCH
        # =====================================================

        step = alpha

        success = False

        res_norm_old = sqrt(
            abs(InnerProduct(res,res))
        )

        for ls in range(12):

            w.data = gfu.vec - step*du

            with TaskManager():

                a.Apply(w, res)

                res_norm_new = sqrt(
                    abs(InnerProduct(res,res))
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
            abs(InnerProduct(du,res))
        )

        print(
            "Newton iteration:",
            it,
            "Residual =",
            res_norm_new,
            "Step =",
            step
        )

        if stopcritval < tol:

            print("Newton converged")

            break

    return gfu, stopcritval, it


# =========================================================
# MAIN
# =========================================================

start_time = datetime.datetime.now()

data = sys.argv

L = float(data[1])

d = float(data[2])

order = int(data[3])

delta_first = int(data[4])

delta_last = int(data[5])

# positive in txt
mu_input = float(data[6])

# internal sign
mu_bar = -mu_input

print("")
print(f'L = {L}')
print(f'd = {d}')
print(f'order = {order}')
print(f'mu_bar = {mu_bar}')

folder_name_suffix = (
    str(int(d))
    + '_'
    + str(int(d%1*100)).zfill(2)
)

delta_values = np.loadtxt(
    'meshes'
    + folder_name_suffix
    + '/deltas'
)

indexes_deltas = range(
    delta_first,
    delta_last+1
)

print(f'indexes_deltas = {indexes_deltas}')

for index_delta in indexes_deltas:

    delta = delta_values[index_delta]

    print("")
    print(
        "Mesh number = {}, delta={:.3f}".format(
            index_delta,
            delta
        )
    )

    gel = gel_debonded2D(
        folder_name_suffix=folder_name_suffix,
        length=L,
        thickness=d,
        delta=delta,
        mu_bar=mu_bar
    )

    modelling = Solve_gel_bonded(
        gel,
        order=order
    )

    modelling.add_mesh(index_delta)

    modelling.Solve_incremental_softening()

    print("")
    print(
        "Time elapsed =",
        datetime.datetime.now()-start_time
    )