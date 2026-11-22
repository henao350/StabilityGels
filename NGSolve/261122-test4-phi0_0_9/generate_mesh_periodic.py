from ngsolve import *
from ngsolve.webgui import Draw
from netgen.occ import *
import os 

# L = 100; d = 20; maxh = 2
L=10; d=1; maxh=d/10


shape = Rectangle(L,d).Face()
shape.edges.Max(X).name = "right"
shape.edges.Min(X).name = "left"
shape.edges.Max(Y).name = "top"
shape.edges.Min(Y).name = "bot"
# shape.edges.Max(Y).maxh = 0.4
shape.edges.Max(Y).maxh = maxh*(1/5)
shape.edges.Max(X).Identify(shape.edges.Min(X), "lr")

mesh = Mesh(
    OCCGeometry(shape, dim=2).GenerateMesh(
        maxh=maxh,
        grading=0.3
    )
)

MESH_FOLDER = 'meshes'
MESH_NAME = 'meshPeriodic.vol'
if not os.path.exists(MESH_FOLDER):
    os.makedirs(MESH_FOLDER)
#
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# mesh_file = os.path.join(BASE_DIR, MESH_FOLDER, MESH_NAME)
mesh_file = os.path.join(MESH_FOLDER, MESH_NAME)
# filename = 'meshes/meshPeriodic.vol'
mesh.ngmesh.Save(mesh_file)
# Draw(mesh)
