'''
voxelfeatures.py

Voxel features describing local area, curvature and orientation of the surface.

Copyright 2016 by Dmitry Yarotsky
'''
import os
import numpy as np
from ctypes import cdll, POINTER, c_float, c_double, c_long, c_int, byref, c_char_p, cast


def initGeomFeatures():
    '''
    Initialize the C-extension computing the voxel features.
    '''
    global getFeaturesCPP, freemeCPP
    path2so = os.path.join(os.path.dirname(__file__), "geomFeatures.so")
    getFeaturesCPP = cdll.LoadLibrary(path2so).get_features_list_wrap
    freemeCPP = cdll.LoadLibrary(path2so).freeme
           
    getFeaturesCPP.argtypes = [
                  c_long,
                  c_long,
                  POINTER(c_double),
                  POINTER(c_long),
                  c_int,
                  POINTER(c_int),
                  POINTER(c_long),
                  POINTER(POINTER(c_float)),
                  POINTER(POINTER(c_int)),
                  c_int,
                  c_int,
                  POINTER(POINTER(c_char_p))]
    
    freemeCPP.argtypes = [POINTER(POINTER(c_float)),
                          POINTER(POINTER(c_int))]  
    
initGeomFeatures() 


def getDataOff(modelpath):
    '''
    Read surface data from the .off file.
    
    The faces described by the .off file must be triangular (general polygons not supported).
    
    Arguments:
        modelpath: path to the .off file
        
    Returns:
        vertA: (Nvertices, 3)-array of vertices
        faceA: (Nfaces, 3)-array of faces 
    '''
    with open(modelpath) as f:
        lineL = f.readlines()
    Nlines = len(lineL)
    head = lineL[:2]
    assert head[0].startswith('OFF')
    if not len(head[0].strip()) == 3: # for some reason, some ModelNet files have no line break after first line 
        Nheadlines = 1
        sizes = [int(n) for n in head[0][3:].strip().split()]
    else:
        Nheadlines = 2
        sizes = [int(n) for n in head[1].strip().split()]
    assert len(sizes) == 3 and sizes[-1] == 0
    vertA = np.genfromtxt(modelpath, dtype='float64', delimiter=' ', skip_header=Nheadlines, skip_footer=sizes[1])
    faceA_ = np.genfromtxt(modelpath, dtype='int64', delimiter=' ', skip_header=Nheadlines+sizes[0])
    assert len(vertA) == sizes[0]
    assert len(faceA_) == sizes[1]
    assert faceA_.shape[1] == 4
    assert Nlines == Nheadlines+sizes[0]+sizes[1]   
    faceA = faceA_[:,1:]
    return vertA, faceA 


def getVoxelFeatures(vertA, faceA, spatialSize, featureList, splitEmpty):
    '''
    Compute a list of voxel features for the given surface.
    
    Voxels are cubes [nx,nx+1]x[ny.ny+1]x[nz,nz+1] with integer nx,ny,nz in the domain 
    [-spatialSize/2,spatialSize/2]x[-spatialSize/2,spatialSize/2]x[-spatialSize/2,spatialSize/2].  
    The surface (more precisely, the array vertA) should be rescaled appropriately before calling the function.
    
    The features are returned only for non-empty voxels, along with the coordinates of these voxels. 
    
    Arguments:
        vertA: (Nvertices, 3)-array of vertices
        faceA: (Nfaces, 3)-array
        spatialSize: (integer) the linear size of the domain
        featureList: the list of features to be computed
        splitEmpty: 0 or 1.
	            If 0, only non-empty voxels will be recorded.
	            If 1, all voxels will be recorded. This case is currently supported only 
                    when spatialSize is a power of 2.
        
    Returns:
        features: the (nSpatialSites, dimFeatPerVoxel) array of feature values
        x, y, z: the (nSpatialSites,)-arrays of x,y,z coordinates of non-empty voxels (vary from 0 to spatialSize-1)
        dimFeatPerVoxel: total dimension of the feature space of a single voxel 
        nSpatialSites: the number of non-empty voxels
    '''
    assert np.max(np.abs(vertA)) <= float(spatialSize)/2, 'spatialSize too small'
    splitEmpty = int(splitEmpty)
    featureL = list(featureList)
    NfeatureKinds = len(featureL)
    
    features_ = POINTER(c_float)()
    xyz = POINTER(c_int)()
    nSpatialSites_ = c_int()
    size_ = c_long() 
      
    featureL_char_p = [c_char_p(feat) for feat in featureL]
    featureL_c = (c_char_p * len(featureL))(*featureL_char_p)
        
    vertA1 = vertA.ravel()
    faceA1 = faceA.ravel()

    getFeaturesCPP(
      len(vertA), 
      len(faceA), 
      vertA1.ctypes.data_as(POINTER(c_double)), 
      faceA1.ctypes.data_as(POINTER(c_long)), 
      spatialSize, 
      byref(nSpatialSites_),
      byref(size_), 
      byref(features_), 
      byref(xyz),
      splitEmpty,
      NfeatureKinds,
      cast(featureL_c, POINTER(POINTER(c_char_p))))
    
    size = size_.value
    nSpatialSites = nSpatialSites_.value  
    dimFeatPerVoxel = size/nSpatialSites
    assert dimFeatPerVoxel*nSpatialSites == size
    xyz1 = np.copy(np.array(xyz[0:3*nSpatialSites]).reshape((nSpatialSites,3)))
    x,y,z = xyz1[:,0], xyz1[:,1], xyz1[:,2] 
    features = np.copy(np.array(features_[0:size]).reshape((nSpatialSites,-1)))   
    freemeCPP(byref(features_), byref(xyz))
    
    return features, x, y, z, dimFeatPerVoxel, nSpatialSites


def checkEdgeParity(faceA):
    ''' Test that each edge belongs to exactly two faces, and has opposite orientations in them.'''
    edgeL0 = set((row[n], row[(n+1)%3]) for row in faceA for n in range(3))
    edgeL1 = set((edge[1], edge[0]) for edge in edgeL0)
    return (len(edgeL0) == 3*len(faceA)) and (len(edgeL0.symmetric_difference(edgeL1)) == 0) 

