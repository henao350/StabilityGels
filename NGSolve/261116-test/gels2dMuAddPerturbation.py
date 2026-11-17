from ClassGel2d import gel2d, Solve_gel_bonded, SolveNonlinearMinProblem
from ngsolve import *
import numpy as np
import datetime
import os

def compute_solution(solver,
                     mu_bar,
                     tol=1e-08,
                     maxits=50,
                     alpha=5e-2,
                     Draw_solution = True,
                     filename = 'gridfunctions/result_gelsMu2D',
                     Save_iterations = False):
    solver.gel.mu_bar.Set(mu_bar)
    solver.model()
    # solver.gfu, _,_ = SolveNonlinearMinProblem(a= solver.a, gfu = solver.gfu, FreeDofs =solver.fes.FreeDofs(), maxits=maxits, tol=tol)
    #
    mesh = solver.mesh; gel = solver.gel
    ###
    a= solver.a
    gfu = solver.gfu
    FreeDofs =solver.fes.FreeDofs()
    ###
    if Save_iterations:
        gfu.Save(filename + f'_Newton_00.gfu')
    ###
    res = gfu.vec.CreateVector()
    du  = gfu.vec.CreateVector()
    for it in range(1,maxits+1):
        #
        a.Apply (gfu.vec, res)
        a.AssembleLinearization (gfu.vec)
        #
        inv = a.mat.Inverse(FreeDofs)
        du.data = - alpha * inv * res
        gfu.vec.data += du
        #
        if Save_iterations:
            gfu.Save(filename + f'_Newton_{str(it).zfill(2)}.gfu')
        #
        if Draw_solution:
            Draw((y+gfu[1])/d, mesh, deformation=gfu)
        # stopping criteria
        stopcritval = sqrt(abs(InnerProduct(du,res)))
        if stopcritval < tol:
            break
    ###
    # solver.gfu.Save(filename +'.gfu')
    # Now print status
    # gfu = solver.gfu; mesh = solver.mesh; gel = solver.gel
    F=Id(2)+Grad(gfu)
    print('Average energy density = ', round(Integrate(gel.W(F), mesh)/(L*d),5))   
    vertex_values = [(y+gfu[1])(mesh(*p)) for p in mesh.ngmesh.Points()]
    print(f'Maximum vertical stretch: ', max(vertex_values)/gel.d)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    #
    timestamp = start_time.strftime("%Y%m%d%H%M")
    simulation_name = 'test' + timestamp
    if not os.path.exists(simulation_name):
        os.makedirs(simulation_name)
    #
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
    print(f'Start time: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'L={L} [mm], d={d} [mm], order={order}, phi0={phi0}, mu_bar={mu_bar}')
    #
    gel = gel2d(L,d,phi0,mu_bar)
    #
    modelling = Solve_gel_bonded (gel, order=order)
    #
    # mesh_file = 'meshes/meshPeriodic.vol'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_file = os.path.join(base_dir, "meshes", "meshPeriodic.vol")
    if os.path.exists(mesh_file):
        print(f'File {mesh_file} does exist :-)')
    else:
        print(f'File {mesh_file} does not exist :-(')

    modelling.add_mesh(mesh_file)
    # Initialize solver
    solver = modelling
    solver.Space()
    solver.model()
    solver.gfu = GridFunction(solver.fes)
    # Línea de código con el cambio principal de esta versión AddPerturbation.py"
    solver.gfu.Load(f'test202611170858/result_gelsMu2D' 
                    + solver.gel.filename_suffix 
                    + '_iter=100.gfu')
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
    # Small initial surface perturbation:
    # u_2 = A*(y/d)*cos(k*x)
    # with A = amp_rel*d and k = 2*pi*n_waves/L.
    n_waves = 5
    amp_rel = 0.01
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
    # -------------------------------------------------
    # Introduce the small surface perturbation.
    # -------------------------------------------------
    print('Adding surface perturbation...')

    solver.gfu.vec.data += perturbation.vec

    # Newton's

    tol = 1e-3
    # maxits = 100
    maxits = 100
    alpha = 5e-2

    mu_bar = -0.05

    filename = f'{simulation_name}/result_gelsMu2D' + \
             solver.gel.filename_suffix + \
             '_withPerturbation'

    compute_solution(
        solver,
        mu_bar,
        tol=tol,
        maxits=maxits,
        alpha=alpha,
        Draw_solution=False,
        filename=filename,
        Save_iterations=True
    )

    print("Time elapsed =" + str(datetime.datetime.now() - start_time))
    print(" ")
