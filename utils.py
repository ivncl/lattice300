#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 22:14:06 2023

@author: ivan
"""

def PoissonDiskSampler(BoundMesh,CellDens,BoundTol,DensTol,MaxAttempts=100000):
    """
    Poisson Disk Sampling algorithm with variable density through a scalar 
    function f(x,y,z)
    """  
    
    import numpy as np
    import trimesh

    # # select first point
    BoundMeshBounds=BoundMesh.bounds
    Lx=BoundMeshBounds[1,0]-BoundMeshBounds[0,0]
    Ly=BoundMeshBounds[1,1]-BoundMeshBounds[0,1]
    Lz=BoundMeshBounds[1,2]-BoundMeshBounds[0,2]
    dist=-1
    while dist<0:
        # generate sample
        FirstPoint=np.array([ \
                             np.random.rand(1)*Lx+BoundMeshBounds[0,0], \
                                 np.random.rand(1)*Ly+BoundMeshBounds[0,1], \
                                     np.random.rand(1)*Lz+BoundMeshBounds[0,2], \
                                         ])     
        FirstPoint=np.reshape(FirstPoint,[1,3])
        dist=trimesh.proximity.signed_distance(BoundMesh, FirstPoint)
        # print(dist)
    print('Point 0: %.2f, %.2f, %.2f' % (FirstPoint[0][0],FirstPoint[0][1],FirstPoint[0][2]))
    SamplePoints=np.zeros((0,3))
    SamplePoints=np.r_[SamplePoints,FirstPoint]

    # iterate
    k=0
    kk=[0]
    mindistbase=np.max(BoundMeshBounds)*1e+10*np.ones(3)
    min_distances=mindistbase
    while k<MaxAttempts:
        k=k+1
        dist=-1
        while dist<0:
            # generate sample
            RandomPoint=np.array([ \
                                 np.random.rand(1)*Lx+BoundMeshBounds[0,0], \
                                     np.random.rand(1)*Ly+BoundMeshBounds[0,1], \
                                         np.random.rand(1)*Lz+BoundMeshBounds[0,2], \
                                             ])     
            RandomPoint=np.reshape(RandomPoint,[1,3])
            dist=trimesh.proximity.signed_distance(BoundMesh, RandomPoint)
        current_min_dist=CellDens(RandomPoint[0][0],RandomPoint[0][1],RandomPoint[0][2])
        dist=[]
        for i in range(0,len(SamplePoints)):
            curr_dist=SamplePoints[i,:]-RandomPoint
            dist.append(np.linalg.norm(curr_dist))
        min_dist=np.min(dist)
        oldmindists=min_distances.copy()
        min_distances=np.append(oldmindists,min_dist)
        if current_min_dist<min_dist:
            SamplePoints=np.r_[SamplePoints,RandomPoint]
            print('Point %d: %.2f, %.2f,% .2f -- Attempts: %d -- Nearest point at %.2f (min %.2f)' % \
                  (len(SamplePoints)-1,SamplePoints[-1,0],SamplePoints[-1,1],SamplePoints[-1,2],k, \
                      min_dist,current_min_dist) )
            kk.append(k)
            k=0; 
            
    return SamplePoints,min_distances

def Mesh2BRep(STLFile):
    """
    STL to BRep in gmsh.
    https://gitlab.onelab.info/gmsh/gmsh/-/issues/1862
    ...
    """  

    import gmsh
    import sys
    import numpy as np
    
    gmsh.initialize()
    
    # load the STL mesh and retrieve the element, node and edge data
    gmsh.open(STLFile)
    typ = 2 # 3-node triangles
    elementTags, elementNodes = gmsh.model.mesh.getElementsByType(typ)
    nodeTags, nodeCoord, _ = gmsh.model.mesh.getNodesByElementType(typ)
    edgeNodes = gmsh.model.mesh.getElementEdgeNodes(typ)    
    # create a new model to store the BREP
    gmsh.model.add('my brep')    
    # create a geometrical point for each mesh node
    nodes = dict(zip(nodeTags, np.reshape(nodeCoord, (len(nodeTags), 3))))
    for n in nodes.items():
        gmsh.model.occ.addPoint(n[1][0], n[1][1], n[1][2], tag=n[0])   
    # create a geometrical plane surface for each (triangular) element
    allsurfaces = []
    allcurves = {}
    elements = dict(zip(elementTags, np.reshape(edgeNodes, (len(elementTags), 3, 2))))
    for e in elements.items():
        curves = []
        for edge in e[1]:
            ed = tuple(np.sort(edge))
            if ed not in allcurves:
                t = gmsh.model.occ.addLine(edge[0], edge[1])
                allcurves[ed] = t
            else:
                t = allcurves[ed]
            curves.append(t)
        cl = gmsh.model.occ.addCurveLoop(curves)
        allsurfaces.append(gmsh.model.occ.addPlaneSurface([cl]))   
    # create a volume bounded by all the surfaces
    sl = gmsh.model.occ.addSurfaceLoop(allsurfaces)
    gmsh.model.occ.addVolume([sl])  
    gmsh.model.occ.synchronize()
    gmsh.write("BoundGeom.brep")   
    # gmsh.fltk.run()   
    gmsh.finalize()

def VOROGenerator(SamplePoints):
    """
    Voronoi struts from Sample Points.
    ...
    """  
    
    from scipy.spatial import Voronoi as voroo  
    import numpy as np      
    voro=voroo(SamplePoints)        
    V=voro.vertices.copy()
    all_E=np.zeros((0,2),dtype='int')
    polygons=voro.ridge_vertices
    for i in range(0,len(polygons)):
        polygons[i].append(polygons[i][0])
        ridges=polygons[i]
        NRidges=len(ridges)
        for j in range(0,NRidges-1):
            # print(i,j,polygons[i])
            # print(polygons[i][j],polygons[i][j+1])
            curr_ridge=np.array([polygons[i][j],polygons[i][j+1]])
            curr_ridge=curr_ridge.reshape(1,2)
            all_E=np.r_[all_E,curr_ridge]
    out_vertices=np.argwhere(all_E==-1)
    out_vertices=out_vertices[:,0]
    all_E=np.delete(all_E,out_vertices,axis=0)
    all_E.sort(axis=1)
    all_E = all_E[all_E[:,0].argsort()]
    all_E=np.unique(all_E,axis=0)
    E=all_E
    
    return V,E

def DELAGenerator(SamplePoints):
    """
    Delaunay struts from Sample Points.
    ...
    """  
    
    import numpy as np      
    from scipy.spatial import Delaunay as delaa
    dela=delaa(SamplePoints)           
    V=dela.points.copy()
    all_E=np.zeros((0,2),dtype='int')
    # get edges
    polygons=dela.simplices
    polygons=np.c_[polygons,polygons[:,0]]
    for i in range(0,len(polygons)):
        # polygons[i].append(polygons[i][0])
        ridges=polygons[i]
        NRidges=len(ridges)
        for j in range(0,NRidges-1):
            # print(i,j,polygons[i])
            # print(polygons[i][j],polygons[i][j+1])
            curr_ridge=np.array([polygons[i][j],polygons[i][j+1]])
            curr_ridge=curr_ridge.reshape(1,2)
            all_E=np.r_[all_E,curr_ridge]  
    out_vertices=np.argwhere(all_E==-1)
    out_vertices=out_vertices[:,0]
    all_E=np.delete(all_E,out_vertices,axis=0)
    all_E.sort(axis=1)
    all_E = all_E[all_E[:,0].argsort()]
    all_E=np.unique(all_E,axis=0)
    E=all_E      
    
    return V,E

def FitInBoundBox(V,E,Bounds):
    """
    Fit geom in bounding cuboid.
    ...
    """  
    
    import numpy as np
    
     # point-normal plane
    # https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
    def isect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6):
        u = p1 - p0
        dot = np.dot(p_no,u)
        if abs(dot) > epsilon:
            w = p0 - p_co
            fac = -np.dot(p_no,w) / dot
            return p0 + (u * fac)
    
        return None
    
    xgrid=[Bounds[0][0],Bounds[1][0]]
    ygrid=[Bounds[0][1],Bounds[1][1]]
    zgrid=[Bounds[0][2],Bounds[1][2]]
    
    # adjust boundaries -- using struts
    points_to_delete=[]
    struts_to_modify=[]
    struts_to_modify_pos=[]
    struts_to_delete=[]
    NewPoints=np.zeros((0,3),dtype='float')
    for i in range(0,len(E)):
        Point0_inside='false'
        Point1_inside='false'
        strut_point_to_cut=None
        Point0_ID=E[i,0]
        Point1_ID=E[i,1]
        Point0=V[Point0_ID]
        Point1=V[Point1_ID]
        if Point0[0]>xgrid[0] and Point0[0]<xgrid[-1] \
            and Point0[1]>ygrid[0] and Point0[1]<ygrid[-1] \
                and Point0[2]>zgrid[0] and Point0[2]<zgrid[-1]:
            Point0_inside='true'
        if Point1[0]>xgrid[0] and Point1[0]<xgrid[-1] \
            and Point1[1]>ygrid[0] and Point1[1]<ygrid[-1] \
                and Point1[2]>zgrid[0] and Point1[2]<zgrid[-1]:
            Point1_inside='true'
        if Point0_inside=='false' and Point1_inside=='true':
                strut_point_to_cut=0
        if Point0_inside=='true' and Point1_inside=='false':
                strut_point_to_cut=1
        # print('strut info:',i,Point0_ID,Point1_ID,Point0,Point1,Point0_inside,Point1_inside,strut_point_to_cut)
        if strut_point_to_cut!=None:
            intersections=np.zeros((0,3),dtype='float')
            midpoint=(Point0+Point1)/2
            p_co_all=np.array([
                [xgrid[0],ygrid[0],zgrid[0]], # xy-down
                [xgrid[-1],ygrid[-1],zgrid[-1]], # xy-up
                [xgrid[0],ygrid[0],zgrid[0]], # xz-front
                [xgrid[0],ygrid[-1],zgrid[0]], # xz-back
                [xgrid[0],ygrid[0],zgrid[0]], # yz-left
                [xgrid[-1],ygrid[0],zgrid[0]], # yz-right
                ])
            p_no_all=np.array([
                [0,0,1], # xy-down
                [0,0,1], # xy-up
                [0,1,0], # xz-front
                [0,1,0], # xz-back
                [1,0,0], # yz-left
                [1,0,0], # yz-right
                ])
            for j in range(0,6):
                p_co=p_co_all[j]
                p_no=p_no_all[j]
                curr_intersection=np.array([isect_line_plane_v3(Point0, Point1, p_co, p_no, epsilon=1e-6)])
                # print('curr int:',curr_intersection)
                try:
                    tol=1e-8
                    if curr_intersection[0][0]>=xgrid[0]-tol and curr_intersection[0][0]<=xgrid[-1]+tol \
                        and curr_intersection[0][1]>=ygrid[0]-tol and curr_intersection[0][1]<=ygrid[-1]+tol \
                            and curr_intersection[0][2]>=zgrid[0]-tol and curr_intersection[0][2]<=zgrid[-1]+tol:
                        intersections=np.r_[intersections,curr_intersection]
                except:
                    pass
            dists_to_midpoint=np.linalg.norm(midpoint-intersections,axis=1)
            # print('all ints, midpoint:',intersections,midpoint)
            # print(midpoint-intersections)
            # print('dists to midpoint:',dists_to_midpoint)
            index=np.argmin(dists_to_midpoint)
            NewPoint=np.array([intersections[index]])
            # print('intersection:',index,intersections[index])
            NewPoints=np.r_[NewPoints,NewPoint]
            struts_to_modify.append(i)
            if strut_point_to_cut==0:
                points_to_delete.append(Point0_ID)
                struts_to_modify_pos.append(0)
            else:
                points_to_delete.append(Point1_ID)
                struts_to_modify_pos.append(1)
        if Point0_inside=='false' and Point1_inside=='false':
            struts_to_delete.append(i)
            points_to_delete.append(Point0_ID)
            points_to_delete.append(Point1_ID)
    # print('params to mod',struts_to_modify,struts_to_modify_pos,points_to_delete,struts_to_delete)            
    VV=V.copy()
    EE=E.copy()           
    offset=len(VV)
    for i in range(0,len(struts_to_modify)):
        strut_id=struts_to_modify[i]
        pos=struts_to_modify_pos[i]
        if pos==0:
            EE[strut_id,0]=i+offset
        else:
            EE[strut_id,1]=i+offset
    # print('elems comp:',EE,E)            
    points_to_delete=np.unique(points_to_delete)
    # print(points_to_delete)
    VVVV=NewPoints
    VV=np.r_[VV,VVVV]                    
    offsets=np.zeros(len(VV),dtype='int')
    # renumbering
    for i in range(0,len(VV)):
        offsets[i]=np.sum(points_to_delete<i)     
    # print(offsets)            
    for i in range(0,len(VV)):
        elements_to_change=np.argwhere(EE==i)
        # print('elems to change:',elements_to_change)
        for j in range(0,len(elements_to_change)):
            indexes=elements_to_change[j]
            index0=indexes[0]
            index1=indexes[1]
            EE[index0,index1]=EE[index0,index1]-offsets[EE[index0,index1]]   
    # print('points:',points_to_delete)
    VVV=np.delete(VV,points_to_delete,axis=0)
    EEE=np.delete(EE,struts_to_delete,axis=0)
                
    SquaredNodes=VVV.copy()
    SquaredElements=EEE.copy()
    
    # nodestoadjust=[]
    # for i in range(0,len(SquaredNodes)):
        # if abs(SquaredNodes[i][0]-0)<.5 or abs(SquaredNodes[i][0]-20)<.5;
            # nodestoadjust.append(i)
    
    return SquaredNodes,SquaredElements

def BuildLattice(V,E,BoundGeom,DiamLaw,meshsize,prefix):
    import gmsh
    import sys
    import numpy as np

    gmsh.initialize()
    gmsh.model.add("boolean")
    ###########################
    """
    STL to BRep in gmsh.
    https://gitlab.onelab.info/gmsh/gmsh/-/issues/1862
    ...
    """  
    # load the STL mesh and retrieve the element, node and edge data
    gmsh.open(BoundGeom)
    typ = 2 # 3-node triangles
    elementTags, elementNodes = gmsh.model.mesh.getElementsByType(typ)
    nodeTags, nodeCoord, _ = gmsh.model.mesh.getNodesByElementType(typ)
    edgeNodes = gmsh.model.mesh.getElementEdgeNodes(typ)    
    # create a new model to store the BREP
    gmsh.model.add('my brep')    
    # create a geometrical point for each mesh node
    nodes = dict(zip(nodeTags, np.reshape(nodeCoord, (len(nodeTags), 3))))
    for n in nodes.items():
        gmsh.model.occ.addPoint(n[1][0], n[1][1], n[1][2], tag=n[0])   
    # create a geometrical plane surface for each (triangular) element
    allsurfaces = []
    allcurves = {}
    elements = dict(zip(elementTags, np.reshape(edgeNodes, (len(elementTags), 3, 2))))
    for e in elements.items():
        curves = []
        for edge in e[1]:
            ed = tuple(np.sort(edge))
            if ed not in allcurves:
                t = gmsh.model.occ.addLine(edge[0], edge[1])
                allcurves[ed] = t
            else:
                t = allcurves[ed]
            curves.append(t)
        cl = gmsh.model.occ.addCurveLoop(curves)
        allsurfaces.append(gmsh.model.occ.addPlaneSurface([cl]))   
    # create a volume bounded by all the surfaces
    sl = gmsh.model.occ.addSurfaceLoop(allsurfaces)
    vol=gmsh.model.occ.addVolume([sl])  
    gmsh.model.occ.synchronize()
    # gmsh.fltk.run()
    ##
    modstruts=[]
    modstrutspoints=[]
    # print(len(E),E)
    for i in range(0,len(E)):
        p1 = gmsh.model.occ.addPoint(V[E[i][0]][0],V[E[i][0]][1],V[E[i][0]][2])
        p2 = gmsh.model.occ.addPoint(V[E[i][1]][0],V[E[i][1]][1],V[E[i][1]][2])
        line = gmsh.model.occ.addLine(p1,p2)
        if i==0:
            firstlinetag=line
        gmsh.model.occ.synchronize()
        outDimTags, outDimTagsMap = gmsh.model.occ.intersect([(1, line)], [(3, vol)], removeTool=False)
        # print(outDimTags)
        if outDimTags!=[]:
            gmsh.model.occ.synchronize()
            entss=gmsh.model.getAdjacencies(1,outDimTags[0][1])
            entsscoords1=gmsh.model.getValue(0,entss[1][0],[])
            entsscoords2=gmsh.model.getValue(0,entss[1][1],[])
            point0=(V[E[i][0]][0]-entsscoords1[0]!=0 or V[E[i][0]][1]-entsscoords1[1] or V[E[i][0]][1]-entsscoords1[2]==0)
            point1=(V[E[i][1]][0]-entsscoords2[0]!=0 or V[E[i][1]][1]-entsscoords2[1] or V[E[i][1]][1]-entsscoords2[2]==0)
            if point0:
                modstruts.append(outDimTags)
                modstrutspoints.append(0)
            if point1:
                modstruts.append(outDimTags)
                modstrutspoints.append(1)
    # print(modstruts)
    gmsh.model.occ.remove([(3,vol)],recursive=True)
    gmsh.model.occ.synchronize()
    ents1=gmsh.model.occ.getEntities(1)
    E=np.zeros((0,2),dtype='int')
    Eids=np.zeros(0,dtype='int')
    V0=np.zeros((0,3),dtype='float')
    V1=np.zeros((0,3),dtype='float')
    for i in range(0,len(ents1)):
        adjs=gmsh.model.getAdjacencies(1,ents1[i][1])
        E=np.r_[E,np.array(np.reshape(adjs[1],(1,2)))]
        Eids=np.r_[Eids,ents1[i][1]]
        V0=np.r_[V0,np.reshape(gmsh.model.getValue(0,adjs[1][0],[]),(1,3))]
        V1=np.r_[V1,np.reshape(gmsh.model.getValue(0,adjs[1][1],[]),(1,3))]
    DD0=[]
    DD1=[]
    Ne=len(E)
    Rangee=range(0,Ne)
    for i in Rangee:
        DD0.append(DiamLaw(V0[i][0],V0[i][1],V0[i][2]))
        DD1.append(DiamLaw(V1[i][0],V0[i][1],V0[i][2]))      
    DD=np.c_[DD0,DD1]
    gmsh.model.occ.remove(ents1,recursive=True)
    np.savetxt(prefix + '_V0', V0, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
    np.savetxt(prefix + '_V1', V1, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
    np.savetxt(prefix + '_E', E, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
    ##############
    ## MESH 3D
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    # gmsh.option.setNumber("Mesh.MaxNumThreads2D", 8)
    # gmsh.option.setNumber("Mesh.MaxNumThreads3D", 8)
    gmsh.option.setNumber("Mesh.MinimumCurveNodes", 1)
    gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize*.8)
    gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize*1.2)
    # gmsh.option.setNumber("Geometry.OCCParallel",1)
    # gmsh.option.setNumber("General.Verbosity", 99)
    # gmsh.option.setNumber('General.Terminal', 0) # don't display messages on the terminal  
    # gmsh.model.add("boolean")
    Ne=len(E)
    Rangee=range(0,Ne)
    cylis = []
    VVVcurr=np.zeros((0,3),dtype='float')
    for i in Rangee:
        DCurr0=DD[i][0]
        DCurr1=DD[i][1]
        # ECurr=E[i]
        VECurr0=V0[i]
        VECurr1=V1[i]
        if DCurr1==DCurr0:
            # build cylinder
            cyl = (3, gmsh.model.occ.addCylinder(VECurr0[0],VECurr0[1],VECurr0[2], \
                                                  VECurr1[0]-VECurr0[0], \
                                                      VECurr1[1]-VECurr0[1], \
                                                          VECurr1[2]-VECurr0[2], \
                                                              DCurr0/2) )
        else:
            # build trunc cone
            cyl = (3, gmsh.model.occ.addCone(VECurr0[0],VECurr0[1],VECurr0[2], \
                                                  VECurr1[0]-VECurr0[0], \
                                                      VECurr1[1]-VECurr0[1], \
                                                          VECurr1[2]-VECurr0[2], \
                                                              DCurr0/2,DCurr1/2) )
        cylis.append(cyl)
        # cond1=np.any((VVVcurr==VECurr0).all(axis=1))
        # cond2=np.any((VVVcurr==VECurr1).all(axis=1))
        # print(len(VVVcurr),cond1,cond2)
        # if cond1==0:
            # sph1 = (3, gmsh.model.occ.addSphere(VECurr0[0],VECurr0[1],VECurr0[2], .51*DCurr0 ) )
            # cylis.append(sph1)
        # if cond2==0:
            # sph2 = (3, gmsh.model.occ.addSphere(VECurr1[0],VECurr1[1],VECurr1[2], .51*DCurr0 ) )
            # cylis.append(sph2)
        # VVVcurr=np.unique(np.r_[np.r_[VVVcurr,np.reshape(VECurr0,(1,3))],np.reshape(VECurr1,(1,3))], axis=0)
    # VVV=np.unique(np.r_[V0,V1], axis=0)
    # print(VVV)
    # Nv=len(VVV)
    # Rangev=range(0,Nv)
    # for i in Rangev:
    #     radius=.51*DiamLaw(VVV[i][0],VVV[i][1],VVV[i][2])
    #     print(radius)
    #     cyl = (3, gmsh.model.occ.addSphere(VVV[i][0],VVV[i][1],VVV[i][2], radius ) )
    #     cylis.append(cyl)
    # Rangeebound=range(0,len(modstruts))
    # for ii in Rangeebound:
    #     cond=np.where(Eids==modstruts[ii][0][1])
    #     i=cond[0][0]
    #     DCurr0=DD[i][0]
    #     DCurr1=DD[i][1]
    #     point=modstrutspoints[ii]
    #     VECurr0=V0[i]
    #     VECurr1=V1[i]
    #     if point==0:
    #         sph = (3, gmsh.model.occ.addSphere(VECurr0[0],VECurr0[1],VECurr0[2], \
    #                                                           DCurr0/2*1.01) )
    #     else:
    #         sph = (3, gmsh.model.occ.addSphere(VECurr1[0],VECurr1[1],VECurr1[2], \
    #                                                           DCurr1/2*1.01) )
    #     cylis.append(sph)    
    # try:
        # outDimTags, outDimTagsMap = gmsh.model.occ.fuse(cylis[:len(cylis)//2],cylis[len(cylis)//2:])
    # except:
        # print('Boolean union or boundary cut failed. :-(')
    gmsh.model.occ.synchronize()
    print('Write geometry.')
    gmsh.write(prefix + 'lattice.brep') 
    gmsh.write(prefix + 'lattice.step') 
    print('Generating and optimizing mesh.')
    # gmsh.model.mesh.generate(3)    
    # gmsh.model.mesh.optimize(method = "Netgen", force = False, niter = 20, dimTags = [] )  
    # print('Write meshes.')
    # gmsh.write(prefix + 'lattice.stl')
    # gmsh.write(prefix + 'lattice.msh')
    # gmsh.write(prefix + 'lattice.key')    
    ##############
    # gmsh.fltk.run()
    gmsh.finalize
    
    # return volume

