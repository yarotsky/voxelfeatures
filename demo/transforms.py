'''
Auxiliary transformations of spatial data.

@author: dmitry.yarotsky
'''

import numpy as np
from numpy.core.umath_tests import matrix_multiply

import voxelfeatures as vf


def randRot(vertA, eps0=1e-4, vertical=False):
    '''
    Rotate the mesh by a random angle from the interval [-eps0, eps0]. 
    If vertical, rotate about the z axis, otherwise choose a random axis.
    '''
    if vertical:
        v0 = np.array([1,0,0])
        v1 = np.array([0,1,0])
        v2 = np.array([0,0,1])
    else:
        v0 = np.random.normal(size=(3,))
        v0 /= np.linalg.norm(v0)    
        v1 = np.random.normal(size=(3,))
        v1 -= v1.dot(v0)*v0
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(v0, v1)
        
    v0, v1, v2 = v0.reshape((3,1)), v1.reshape((3,1)), v2.reshape((3,1))
    
    eps = eps0*(2*np.random.rand()-1)
    
    rotM = (v0.dot(np.cos(eps)*v0.T+np.sin(eps)*v1.T)+
            v1.dot(np.cos(eps)*v1.T-np.sin(eps)*v0.T)+
            v2.dot(v2.T))
    assert np.max(np.abs(rotM.dot(rotM.T)-np.eye(3))) < 1e-8
    
    vertA0 = vertA.dot(rotM)
    
    return vertA0


def pcaSurface(vertA, faceA):
    '''
    PCA of a surface with a constant density (mass proportional to the area).
    
    Returns:
        com: center of mass
        w: principal directions
        U: rotation matrix
    '''
    faceV = np.zeros((len(faceA), 3, 3))
    for n in range(3):
        faceV[:, n, :] = vertA[faceA[:,n]]
    centers = np.mean(faceV, axis=1)
    areas2 = np.linalg.norm(np.cross(faceV[:,1]-faceV[:,0], faceV[:,2]-faceV[:,0]), axis=1)
    com = np.dot(areas2,centers)/(np.sum(areas2))
    
    faceV0 = faceV-com
    # Covariance matrix of triangle vith vertices V = (v1, v2, v3)^t equals aV^tSV, where a is triangle's area
    S = np.array([[2,1,1],
                  [1,2,1],
                  [1,1,2]])/24. 
    Q = np.tensordot(areas2, matrix_multiply(faceV0.transpose((0,2,1)), np.dot(S, faceV0).transpose((1,0,2))), axes=(0,0))
    w, U = np.linalg.eigh(Q)
    return com, w, U


def fitToCube(vertA, spatialSize, mode='scaleShift', faceA=None, randShift=False):
    '''
    Fit a mesh to the cube.
    
    mode='scaleShift': Only rescale and shift (no rotations). If randShift, also shift randomly within the domain.
    mode='pcaRotation': Perform PCA, rotate to principal directions, shift to place COM at the origin, rescale.
    
    '''
    if mode == 'scaleShift':
        mmax = np.max(vertA, axis=0, keepdims=True)
        mmin = np.min(vertA, axis=0, keepdims=True)
        maxdiff = np.max(mmax-mmin)     
        vertA0 = 0.9*spatialSize*(vertA-(mmax+mmin)/2)/maxdiff  
        
        if randShift:
            UB = np.min(float(spatialSize)/2-vertA0, axis=0)
            LB = np.max(-float(spatialSize)/2-vertA0, axis=0)
            assert all(UB >= 0) and all(LB <= 0)
            shift = LB+np.random.rand(3)*(UB-LB)
            vertA0 += shift
                
    elif mode == 'pcaRotation':
        assert not randShift 
        com, _, U = pcaSurface(vertA, faceA)
        vertA0 = (vertA-com).dot(U)
        vertA0 *= 0.9*spatialSize/2/np.max(np.abs(vertA0))
        
    assert np.min(vertA0) >= -float(spatialSize)/2 and np.max(vertA0) <= float(spatialSize)/2
            
    return vertA0


def getHaarMatrix(dim):
    m = int(np.log2(dim))
    assert np.abs(2**m-dim) < 1e-5 and m >= 0
    haar = np.zeros((dim, dim))
    haar[0] = np.ones((dim,))/np.sqrt(dim)
    for n in range(m):
        l = 2**(m-n-1)
        for k in range(2**n):
            haar[2**n+k, k*(2*l):k*(2*l)+l] = np.ones((l,))
            haar[2**n+k, k*(2*l)+l:(k+1)*(2*l)] = -np.ones((l,)) 
            haar[2**n+k] /= 2**((m-n)/2.)
    diff = np.sum(np.abs(haar.dot(haar.T)-np.eye(dim)))
    assert diff < 1e-10
    return haar
