'''
Test that all features are working and time their evaluation.

@author: dmitry.yarotsky
'''

import numpy as np
import time

import voxelfeatures as vf

import transforms 

featureL = ['Bool', 
            'ScalarArea', 'SA', 
            'AreaNormal', 'AN', 
            'QuadForm', 'QF',
            'EigenValues', 'EV',
            'VertexAngularDefect', 'VAD',
            'EdgeAngularDefect', 'EAD',
            'VolumeElement', 'VE',
            ]

vertA, faceA = vf.getDataOff('../datasets/ESB/Solid Of Revolution/90 degree elbows/1309429.off')

spatialSize = 128
print '*** Individual features at spatialSize ', spatialSize
vertA0 = transforms.fitToCube(vertA, spatialSize=spatialSize)

for feat in featureL:
    print '=============', feat
    t0 = time.time()
    features, x, y, z, nFeatPerVoxel, nSpatialSites = vf.getVoxelFeatures(vertA0, faceA, spatialSize, 
                            set([feat]), 0)
    t1 = time.time()
    print 'Time:', t1-t0  
    print 'Features per voxel:', nFeatPerVoxel  

    if feat == 'VertexAngularDefect':
        EulerChar = np.sum(features, axis=0)/(2*np.pi)
        assert np.abs(EulerChar-int(EulerChar)) < 1e-5
        print 'Euler characteristic:', int(EulerChar)  
    
    elif feat == 'AreaNormal':
        print 'Total sum approximately vanishes:', np.linalg.norm(np.sum(features, axis=0)) < 1e-5
         

for k in range(9):
    spatialSize = int(2**k)
    vertA0 = transforms.fitToCube(vertA, spatialSize=spatialSize)
    print '*** All features at spatial size ', spatialSize 
    t0 = time.time()
    features, x, y, z, nFeatPerVoxel, nSpatialSites = vf.getVoxelFeatures(vertA0, faceA, spatialSize, 
                            set(featureL), 0)
    t1 = time.time()
    print 'Time:', t1-t0  
    print 'Features per voxel:', nFeatPerVoxel  
    print 'Number of nonempty voxels:', nSpatialSites 
    print 'Share of non-empty voxels:', float(nSpatialSites)/(spatialSize*spatialSize*spatialSize)          

    
