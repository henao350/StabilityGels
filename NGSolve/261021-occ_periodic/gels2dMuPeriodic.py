from ClassGel2d import gel2d, Solve_gel_bonded, SolveNonlinearMinProblem
from ngsolve import *
import numpy as np
import datetime
import os


def compute_solution(
    solver,
    mu_bar,
    tol=1e-08,
    maxits=50,
    alpha=5e-2,
    Draw_solution=True,
    filename='gridfunctions/result_gelsMu2D'
):
    """
    Solve the nonlinear equilibrium problem at the prescribed chemical
    potential, using the current GridFunction as the initial guess.

    IMPORTANT:
    The perturbation is introduced in the GridFunction, not in the mesh.
    """

    solver.gel.mu_bar.Set(mu_bar)
    solver.model()

    solver.gfu, stopcritval, lastit = SolveNonlinearMinProblem(
        a=solver.a,
        gfu=solver.gfu,
        FreeDofs=solver.fes.FreeDofs(),
        maxits=maxits,
        tol=tol,
        alpha=alpha
    )

    print(f'Newton iterations = {lastit+1}, stopping criterion = {stopcritval}')

    solver.gfu.Save(filename + '.gfu')

    # Now print status
    gfu = solver.gfu
    mesh = solver.mesh
    gel = solver.gel

    F = Id(2) + Grad(gfu)

    energy_density = Integrate(gel.W(F), mesh) / (L*d)
    print('Average energy density = ', energy_density)

    vertex_values = [(y+gfu[1])(mesh(*p)) for p in mesh.ngmesh.Points()]
    print('Maximum vertical stretch: ', max(vertex_values)/gel.d)

    if Draw_solution:
        Draw((y+gfu[1])/d, mesh, deformation=gfu)

    return stopcritval, lastit


if __name__ == '__main__':

    start_time = datetime.datetime.now()

    # ---------------------------------------------------------
    # SIMULATION PARAMETERS
    # ---------------------------------------------------------
    nIterations = 100

    # Small initial surface perturbation:
    # u_2 = A*(y/d)*cos(k*x)
    # with A = amp_rel*d and k = 2*pi*n_waves/L.
    n_waves = 5
    amp_rel = 0.01

    # ---------------------------------------------------------
    # READ PARAMETERS
    # ---------------------------------------------------------
    # Arguments:
    #   1: length L
    #   2: thickness d
    #   3: FE order
    #   4: initial polymer volume fraction phi0
    #   5: absolute value of nondimensional chemical potential
    #
    # Example:
    #   python gels2dMuPeriodic.py 100 10 3 0.7 0.0029
    #
    data = sys.argv

    L = float(data[1])
    d = float(data[2])
    order = int(data[3])
    phi0 = float(data[4])
    mu_input = float(data[5])

    mu_bar = -mu_input

    print(
        f'L={L} [mm], d={d} [mm], order={order}, '
        f'phi0={phi0}, mu_bar={mu_bar}'
    )

    # ---------------------------------------------------------
    # GEL MODEL
    # ---------------------------------------------------------
    gel = gel2d(L, d, phi0, mu_bar)

    modelling = Solve_gel_bonded(gel, order=order)

    mesh_file = 'meshes/meshPeriodic.vol'
    modelling.add_mesh(mesh_file)

    solver = modelling
    solver.Space()
    solver.model()

    solver.gfu = GridFunction(solver.fes)
    solver.gfu.vec[:] = 0

    # ---------------------------------------------------------
    # CONTINUATION IN SWELLING
    # ---------------------------------------------------------
    mu_bar_end = solver.gel.mu_bar.Get()
    print(f'mu_bar_end = {mu_bar_end}')

    # Kang-Huang-type numerical continuation starts from
    # a homogeneous swollen state around lambda_h = 2.
    lambda_initial = 2.0

    lambda_list = np.linspace(
        lambda_initial,
        solver.gel.lambda_target,
        nIterations+1,
        endpoint=True
    )

    mu_bar_list = [
        solver.gel.mu_fun(la)
        for la in lambda_list[:-1]
    ]
    mu_bar_list.append(mu_bar_end)

    # ---------------------------------------------------------
    # OUTPUT DIRECTORY
    # ---------------------------------------------------------
    simulation_name = (
        f'test_periodic_L{int(L)}_d{int(d)}'
        f'_phi0={phi0}_nIterations{nIterations}'
    )

    if not os.path.exists(simulation_name):
        os.makedirs(simulation_name)

    # Save continuation table
    filename = f'{simulation_name}/lambda_uniaxial_mu_bar_list.txt'

    with open(filename, 'w') as f:
        f.write(f'nIterations = {nIterations}\n')
        f.write(f'n_waves = {n_waves}\n')
        f.write(f'amp_rel = {amp_rel}\n')
        f.write('lambda_uniaxial, mu_bar\n')

        for la, mu in zip(lambda_list, mu_bar_list):
            f.write(f'{la}, {mu}\n')

    # ---------------------------------------------------------
    # INITIAL SURFACE PERTURBATION
    # ---------------------------------------------------------
    #
    # DO NOT perturb the mesh.
    #
    # The reference configuration remains the flat rectangle.
    # The perturbation is introduced in the displacement field:
    #
    #     u_2(x,y) = A*(y/d)*cos(k*x)
    #
    # so that u_2(x,0)=0 at the rigid substrate and the surface
    # amplitude is A.
    #
    A = amp_rel * d
    k = 2*np.pi*n_waves/L

    perturbation = GridFunction(solver.fes)

    perturbation.Set(
        (
            0,
            A*(y/d)*cos(k*x)
        )
    )

    print('---------------------------------------------')
    print('Initial surface perturbation')
    print(f'n_waves = {n_waves}')
    print(f'wavelength = {L/n_waves}')
    print(f'A = {A}')
    print(f'A/d = {amp_rel}')
    print(f'k = {k}')
    print(f'k*d = {k*d}')
    print('---------------------------------------------')

    # ---------------------------------------------------------
    # CONTINUATION
    # ---------------------------------------------------------
    indexes_iterations = range(nIterations+1)

    for numIteration in indexes_iterations:

        lambda_i = lambda_list[numIteration]
        mu_bar_i = mu_bar_list[numIteration]

        print()
        print(
            f'Iteration {numIteration}: '
            f'lambda_uniaxial={lambda_i}, '
            f'mu_bar={mu_bar_i}'
        )

        tol = 1e-3
        maxits = 100
        alpha = 5e-2

        filename = (
            f'{simulation_name}/result_gelsMu2D'
            + solver.gel.filename_suffix
            + '_iter='
            + str(numIteration).zfill(2)
        )

        # -----------------------------------------------------
        # First solve the homogeneous state at lambda = 2.
        # -----------------------------------------------------
        if numIteration == 0:

            homogeneous_filename = filename + '_homogeneous'

            compute_solution(
                solver,
                mu_bar_i,
                tol=tol,
                maxits=maxits,
                alpha=alpha,
                Draw_solution=False,
                filename=homogeneous_filename
            )

            # -------------------------------------------------
            # Introduce the small surface perturbation.
            # -------------------------------------------------
            print('Adding surface perturbation...')

            solver.gfu.vec.data += perturbation.vec

            # Re-equilibrate the perturbed state at the same
            # chemical potential.
            compute_solution(
                solver,
                mu_bar_i,
                tol=tol,
                maxits=maxits,
                alpha=alpha,
                Draw_solution=False,
                filename=filename
            )

        else:

            # Continue from the previously converged,
            # perturbed/deformed solution.
            compute_solution(
                solver,
                mu_bar_i,
                tol=tol,
                maxits=maxits,
                alpha=alpha,
                Draw_solution=False,
                filename=filename
            )

    print()
    print(
        'Time elapsed = '
        + str(datetime.datetime.now() - start_time)
    )
    print(' ')
