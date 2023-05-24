#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 21:58:04 2023

@author: ivan

Module for generate grid and connectivity for a generic lattice box.

"""

class VORODELALattice:
    
    """
    Voronoi/Delaunay strut lattice class.
    ATTRIBUTES:
        INPUTS:
            - point cloud --> point positions
            - cell density f(x,y,z)
            - strut diameter f(x,y,z)
            - bounding geometry
                - as .stl file
            - Name
        OUTPUTS:
            - surface mesh: .stl
            - solid mesh: .msh, .key
    """
    
    def __init__(self, PointCloud, BoundGeomFile, CellDens, DiamLaw, meshsize, \
                 VORO=True, DELA=True, **kw):
        
        import numpy as np
        import datetime
        import trimesh
        from utils import PoissonDiskSampler, Mesh2BRep, VOROGenerator, DELAGenerator, \
            FitInBoundBox, BuildLattice
        
        BoundMesh = trimesh.load(BoundGeomFile)
        Bounds=BoundMesh.bounds
        # print(Bounds)
        BoundTol=.1
        DensTol=.1
        ## sample
        self.SamplePoints, self.min_distances= \
            PoissonDiskSampler(BoundMesh,CellDens,BoundTol,DensTol)
        np.savetxt('SamplePoints', self.SamplePoints, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
        ## create geometry
        if VORO==True:
            V, E = \
                VOROGenerator(self.SamplePoints)
            ## fit geometry to boundaries and mesh
            BuildLattice(V,E,BoundGeomFile,DiamLaw,meshsize,prefix='VORO')
        if DELA==True:
            V, E = \
                DELAGenerator(self.SamplePoints)
            ## fit geometry to boundaries and mesh
            BuildLattice(V,E,BoundGeomFile,DiamLaw,meshsize,prefix='DELA')
