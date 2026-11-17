from ClassGel2d import gel2d, Solve_gel_bonded, SolveNonlinearMinProblem
from ngsolve import *
import numpy as np
import datetime
import os

def compute_solution(solver, mu_bar, tol=1e-08, maxits=50, alpha=5e-2, Draw_solution = True, filename = 'gridfunctions/result_gelsMu2D'):
    solver.gel.mu_bar.Set(mu_bar)
    solver.model()
    solver.gfu, _,_ = SolveNonlinearMinProblem(a= solver.a, gfu = solver.gfu,\
            FreeDofs =solver.fes.FreeDofs(), maxits=maxits, tol=tol)
    solver.gfu.Save(filename +'.gfu')
    # Now print status
    gfu = solver.gfu; mesh = solver.mesh; gel = solver.gel
    F=Id(2)+Grad(gfu)
    print('Average energy density = ', round(Integrate(gel.W(F), mesh)/(L*d),5))   
    vertex_values = [(y+gfu[1])(mesh(*p)) for p in mesh.ngmesh.Points()]
    print(f'Maximum vertical stretch: ', max(vertex_values)/gel.d)
    if Draw_solution:
        Draw((y+gfu[1])/d, mesh, deformation=gfu)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    #
    timestamp = start_time.strftime("%Y%m%d%H%M")
    simulation_name = 'test' + timestamp
    # simulation_name = 'test202611151635_mesh_grading_absMuBar_0.001_d20'
    # simulation_name = 'test4_periodic_L100_d20_nIterations100'
    # simulation_name = 'test3_periodic_L100_d20_nIterations1'
    # simulation_name = 'test2_dirichletx_L100_d20_nIterations1'
    # simulation_name = 'test1_periodic_L100_d20_nIterations2'    
    
    # if folder {simulation_name} does not exist, create it
    if not os.path.exists(simulation_name):
        os.makedirs(simulation_name)
    nIterations = 100
    nIterationsEffective = 100
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
    mesh_file = 'meshes/meshPeriodic.vol'
    modelling.add_mesh(mesh_file)
    # Initialize solver
    solver = modelling
    solver.Space()
    solver.model()
    solver.gfu = GridFunction(solver.fes)
    solver.gfu.vec[:] = 0
    # solver.gfu.Load(f'test202611170112/result_gelsMu2D' + solver.gel.filename_suffix + '_iter=57.gfu')
    #
    mu_bar_end = solver.gel.mu_bar.Get()
    print(f'mu_bar_end = {mu_bar_end}')
    lambda_initial = 1.01
    lambda_list = np.linspace(lambda_initial, solver.gel.lambda_target, nIterations+1, endpoint = True)
    mu_bar_list = [solver.gel.mu_fun(la) for la in lambda_list[:-1]]
    mu_bar_list.append(mu_bar_end)
    # now save this list in a text file
    filename = f'{simulation_name}/lambda_uniaxial_mu_bar_list.txt'
    with open(filename, 'w') as f:
        f.write(f'nIterations = {nIterations}\n')
        f.write('lambda_uniaxial, mu_bar\n')
        for la, mu in zip(lambda_list, mu_bar_list):
            f.write(f'{la}, {mu}\n')

    # indexes_iterations = range(nIterations+1)
    # nIterationsEffective = 100
    # nIterationsEffective = nIterations
    indexes_iterations = range(nIterationsEffective+1)
    for numIteration in indexes_iterations:
        lambda_i = lambda_list[numIteration]
        mu_bar_i = mu_bar_list[numIteration]
        print(f'Iteration {numIteration}: lambda_uniaxial={lambda_i}, mu_bar={mu_bar_i}')
        tol = 1e-3; maxits = 100;
        filename = f'{simulation_name}/result_gelsMu2D' + solver.gel.filename_suffix + '_iter=' + str(numIteration).zfill(2)
        print(f'filename={filename}')
        compute_solution(solver, mu_bar_i, tol=tol, maxits=maxits, alpha=5e-2, Draw_solution = False, filename = filename)

    #
    print("Time elapsed =" + str(datetime.datetime.now() - start_time))
    print(" ")

