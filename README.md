Python code to generate geometry files and high-quality tetrahedral meshes of Voronoi and Delaunay 3D reticula. 

Supported lattices are now stochastic Voronoi and Delaunay "perfect" reticula. 
Sampling is computed through the Poisson Disk Algorithm, adapted to support arbitrary density functions of minimum distance. Arbitrary function for strut diameter is supported as well. The gometries may be bounded by any shape (to be imported as .stl file), but to date a cuboid is the most efficient.

If using this code, especially for research purposes, kindly cite the [related scientific paper](https://www.sciencedirect.com/science/article/pii/S0020768323003980).

Dependencies:
gmsh
trimesh
scipy
numpy
datetime
sys

Follows example.


```
def MyCellDens(x,y,z):
    return 5
def MyDiamLaw(x,y,z):
    return .8

from lattice300 import VORODELALattice
import gmsh

## create bound box for PDS
gmsh.initialize()
x0, y0, z0 = 0, 0, 0
lx, ly, lz = 20, 20, 20
box = gmsh.model.occ.addBox(x0, y0, z0, lx, ly, lz)
mesh_size = 0.2
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate()
gmsh.write("boxPDS.stl")
gmsh.finalize()
meshsize=.2
lattice=VORODELALattice([], "boxPDS.stl", MyCellDens, MyDiamLaw, meshsize, DELA='False')
```
