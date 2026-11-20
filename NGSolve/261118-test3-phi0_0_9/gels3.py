# from ClassGels3 import gel2d, Solve_gel_bonded, SolveNonlinearMinProblem
from ClassGels3 import gel2d
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
                     filename = 'gridfunctions/result_gelsMu2D'):
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
    print(f'Macroscopic vertical stretch: ', max(vertex_values)/gel.d)
    if Draw_solution:
        Draw(solver.gel.W(F), mesh, deformation=gfu)

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

    gel = gel2d(L,d,phi0,lambda_target)
