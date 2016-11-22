'''
Created on Nov 10, 2016

@author: dmitry.yarotsky
'''

import numpy as np
import time

import voxelfeatures as vf

import transforms

featureL = ['Bool', 
            'ScalarArea', 
            'AreaNormal', 
            'QuadForm', 
            'EigenValues', 
            'VertexAngularDefect', 
            'EdgeAngularDefect', 
            'VolumeElement', 
            ]

vertA, faceA = vf.getDataOff('../datasets/ESB/Solid Of Revolution/90 degree elbows/1309429.off')

for k in range(7):
    spatialSize = int(2**k)
    vertA0 = transforms.fitToCube(vertA, spatialSize=spatialSize)
    print '============ Spatial size ', spatialSize 
    print '*** Sparse voxels'
    t0 = time.time()
    features, x, y, z, nFeatPerVoxel, nSpatialSites = vf.getVoxelFeatures(vertA0, faceA, spatialSize, 
                            set(featureL), 0)
    t1 = time.time()
    print 'Time:', t1-t0  
    print 'Voxels:', nSpatialSites
    print 'Share of filled voxels:', float(nSpatialSites)/spatialSize**3
    
    print '*** Dense voxels'
    t0 = time.time()
    features, x, y, z, nFeatPerVoxel, nSpatialSites = vf.getVoxelFeatures(vertA0, faceA, spatialSize, 
                            set(featureL), 1)
    t1 = time.time()
    print 'Time:', t1-t0  
    print 'Voxels:', nSpatialSites
    print 'Share of filled voxels:', float(nSpatialSites)/spatialSize**3
