from ngsolve import *
from ngsolve.webgui import Draw
from netgen.occ import *

L = 100
d = 20
maxh = 2

shape = Rectangle(L,d).Face()
shape.edges.Max(X).name = "right"
shape.edges.Min(X).name = "left"
shape.edges.Max(Y).name = "top"
shape.edges.Min(Y).name = "bot"
shape.edges.Max(Y).maxh = 0.4
shape.edges.Max(X).Identify(shape.edges.Min(X), "lr")

mesh = Mesh(
    OCCGeometry(shape, dim=2).GenerateMesh(
        maxh=maxh,
        grading=0.3
    )
)

filename = 'meshes/meshPeriodic.vol'
mesh.ngmesh.Save(filename)
Draw(mesh)
