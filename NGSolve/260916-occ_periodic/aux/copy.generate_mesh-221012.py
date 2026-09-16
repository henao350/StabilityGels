import netgen.geom2d as geom2d
from ngsolve import *
#from ngsolve.webgui import Draw
import netgen.gui
import numpy as np



def geometry_partially_bonded(numberofdeltas= 4, number_bonded_intervals=4):
    L = 90      # measured in mm
    d = 15    # measured in mm
    maxh = d/(4-0.1)
    # Points:
    deltas = np.linspace(0.03,0.97,numberofdeltas).tolist()
    np.savetxt('220927-nested/meshes15/deltas', deltas)
    print(deltas)
    nd = len(deltas)
    points = []
    points.append([-0.5*L,0])
    print(deltas)
    print( list(reversed(deltas)))
    for delta in list(reversed(deltas)):
        points.append([-0.5*L*delta,0])
    for delta in deltas:
        points.append([0.5*L*delta,0])
    points.append([0.5*L,0])
    points.append([0.5*L, d])
    points.append([-0.5*L, d])
    print(points)
    geo = geom2d.SplineGeometry()
      
    ps = [geo.AddPoint (x,y) for x,y in points ]
    for i in range(len(ps)-1):
        if i<2*nd+1: # bottom face
            if abs(nd-i)>number_bonded_intervals: #debonded
                geo.Append(["line",ps[i],ps[i+1]],bc='debonded_interface')
                print("debonded:", [ps[i],ps[i+1]])
            else:
                geo.Append(["line",ps[i],ps[i+1]],bc='bonded_interface')
                print("bonded:", [ps[i],ps[i+1]])
            # end
        else:
            geo.Append(["line",ps[i],ps[i+1]])
    geo.Append(["line",ps[len(ps)-1],ps[0]])
    Draw(geo)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    Draw(mesh)
    filename = '220927-nested/meshes15/mesh'+str(number_bonded_intervals)+'.vol'
    mesh.ngmesh.Save(filename)


numberofdeltas = 100
for i in range(0,numberofdeltas):
    geometry_partially_bonded(numberofdeltas, number_bonded_intervals=i)