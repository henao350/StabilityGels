# from ClassGels3 import gel2d, Solve_gel_bonded, SolveNonlinearMinProblem
from ClassGels3 import gel2d, Solve_gel_bonded, SolveNonlinearMinProblem
from ngsolve import *
import numpy as np
import datetime
import os

# def compute_solution(solver,
#                      mu_bar,
#                      tol=1e-08,
#                      maxits=50,
#                      alpha=5e-2,
#                      Draw_solution = True,
#                      simulation_name = 'test2026',
#                      Save_iterations = False
#                      ):
#     solver.gel.mu_bar.Set(mu_bar)
#     solver.model()
#     # solver.gfu, _,_ = SolveNonlinearMinProblem(a= solver.a, gfu = solver.gfu,\
#     #         FreeDofs =solver.fes.FreeDofs(), maxits=maxits, tol=tol, alpha=alpha)
#     solver.gfu.Save(filename +'.gfu')
#     # Now print status
#     gfu = solver.gfu; mesh = solver.mesh; gel = solver.gel
#     F=Id(2)+Grad(gfu)
#     print('Average energy density = ', round(Integrate(gel.W(F), mesh)/(L*d),5))   
#     vertex_values = [(y+gfu[1])(mesh(*p)) for p in mesh.ngmesh.Points()]
#     print(f'\'Macroscopic\' vertical stretch: ', max(vertex_values)/gel.d)
#     if Draw_solution:
#         Draw((y+gfu[1])/d, mesh, deformation=gfu)
#         Draw(solver.gel.W(F), mesh, deformation=gfu)
#         Draw(-solver.gel.sigma_xx(F), mesh, deformation=gfu)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    #
    timestamp = start_time.strftime("%Y%m%d%H%M")
    simulation_name = 'test' + timestamp
    # if folder {simulation_name} does not exist, create it
    if not os.path.exists(simulation_name):
        os.makedirs(simulation_name)
    #### READ PARAMETERS ####
    # It will be assumed that the arguments passed from shell
    # are, in that order:
    # the length; the thickness; the order for the finite element space;
    # the initial polymer volume fraction phi0; 
    # the target theoretical uniaxial stretch lambda_h 
    # For example,
    #    100 20 3 0.9 2.0
    # means L=100, d=20, order=3, phi0=0.9, lambda_h=2.0
    data = sys.argv
    L = float(data[1])
    d = float(data[2])
    order = int(data[3])
    phi0 = float(data[4])
    lambda_target = float(data[5])
    print(f'Start time: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'L={L} [mm], d={d} [mm], order={order}, phi0={phi0}, lambda_target={lambda_target}')
    #
    gel = gel2d(L,d,phi0,lambda_target)
    #
    solver = Solve_gel_bonded (gel, order=order)
    #
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # mesh_file = os.path.join(base_dir, "meshes", "meshPeriodic.vol")
    mesh_file = os.path.join("meshes", "meshPeriodic.vol")
    if os.path.exists(mesh_file):
        print(f'File {mesh_file} does exist :-)')
    else:
        print(f'File {mesh_file} does not exist :-(')
    solver.add_mesh(mesh_file)
    ###
    # Initialize solver
    ###
    # solver = modelling
    solver.Space()
    solver.model()
    solver.gfu = GridFunction(solver.fes)
    # solver.gfu.Load(f'test202611170112/result_gelsMu2D' + solver.gel.filename_suffix + '_iter=57.gfu')
    # ---------------------------------------------------------
    # INITIAL GUESS
    # Set the homogeneous uniaxial swelling,
    # with stretch lambda_target,
    # as initial guess for the Newton iterations.
    # ---------------------------------------------------------
    # solver.gfu.vec[:] = 0
    solver.gfu.Set(
        (
            0,
            (lambda_target-1) * y
        )
    )
    # ---------------------------------------------------------
    # SETS DAMPING PARAMETER FOR ALL THE INSTANCES WHERE NEWTON'S METHOD WILL BE CALLED
    # ---------------------------------------------------------
    alpha=5e-2
    # ---------------------------------------------------------
    # NEWTON ITERATIONS FOR THE FIRST SOLUTION
    # ---------------------------------------------------------
    tol = 1e-3; maxits = 100; Save_iterations = False
    solver.gfu,_,_ = SolveNonlinearMinProblem(a=solver.a, gfu = solver.gfu, FreeDofs =solver.fes.FreeDofs(), tol=tol, maxits=maxits, alpha=alpha, Save_iterations=Save_iterations)
    # ---------------------------------------------------------
    # INITIAL SURFACE PERTURBATION
    # Now a perturbation is added to the solution, and the Newton iterations are performed again.
    # ---------------------------------------------------------
    # The perturbation is introduced in the displacement field:
    #     u_2(x,y) = A*(y/d)*cos(k*x)
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
    #
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
    #
    solver.gfu.vec.data += perturbation.vec
    #
    tol=1e-6; maxits=500; Save_iterations = False
    solver.gfu,_,_ = SolveNonlinearMinProblem(a=solver.a, gfu = solver.gfu, FreeDofs =solver.fes.FreeDofs(), tol=tol, maxits=maxits, alpha=alpha, Save_iterations=Save_iterations)
    # #
    # # ---------------------------------------------------------
    # # RAISE THE CHEMICAL POTENTIAL AND APPLY NEWTON'S METHOD AGAIN
    # # ---------------------------------------------------------
    # new_mu_bar= -0.040
    # solver.update_chemical_potential(new_mu_bar=new_mu_bar)
    # tol=1e-6; maxits=500; Save_iterations = True
    # filename_main = solver.gel.filename_main(simulation_name)
    # solver.gfu,_,_ = SolveNonlinearMinProblem(a=solver.a, gfu = solver.gfu, FreeDofs =solver.fes.FreeDofs(), tol=tol, maxits=maxits, alpha=alpha, Save_iterations=Save_iterations, filename_main=filename_main)