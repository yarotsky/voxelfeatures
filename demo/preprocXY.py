'''
Preprocessing of the ESB or ModelNet data into array-like form, for training and testing classification models. 

@author: dmitry.yarotsky
'''

import os
import multiprocessing
import itertools
import numpy as np
import pandas as pd
import scipy.sparse as sparse

from sklearn.preprocessing import LabelEncoder

import voxelfeatures as vf
import transforms 


def getXsingle(rep, seed, DF, label, dataSettings):
    S = dataSettings
    np.random.seed(seed)
    X = [] 
    haarD = {} 
    for feat in S['featureL']:
        if 'haar' in feat[2] and not feat[0] in haarD.keys():  
            haarD[feat[0]] = transforms.getHaarMatrix(feat[0])
    for n in range(len(DF[label])):
        path = os.path.join(S['rootPath'], DF[label].iloc[n,0])        
        vertA, faceA = vf.getDataOff(path)
        if S['randRotAngle']:
            vertA = transforms.randRot(vertA, eps0=S['randRotAngle'], vertical=S['vertical'])
        if S['modeFitToCube'] in ['pca', 'pca_mult']:           
            vertA = transforms.fitToCube(vertA, 1, mode='pcaRotation', faceA=faceA, randShift=False)                
            m, s = rep%8, rep//8
            reflect = (m//4, (m%4)//2, m%2) 
            perm = list(itertools.permutations(range(3)))[s]
            vertA = vertA[:, perm]
            for d in range(3):
                vertA[:,d] *= 2*reflect[d]-1 
        elif S['modeFitToCube'] == 'scaleShift':
            vertA = transforms.fitToCube(vertA, 1, mode=S['modeFitToCube'], randShift=S['randShift'])
        else:
            raise NotImplementedError
        fL = []
        for feat_ in S['featureL']:
            spatialSize = feat_[0]
            vertA0 = spatialSize*vertA
            
            features, x, y, z, nFeat, _ = vf.getVoxelFeatures(
                                            vertA0, faceA, spatialSize, feat_[1], 0)
            
            inds = x+spatialSize*y+spatialSize*spatialSize*z 
            data = []
            for s in range(nFeat): 
                for postproc in feat_[2]:
                    if postproc == 'voxel':
                        data_ = sparse.csr_matrix((features[:,s], (np.zeros_like(inds), inds)), shape=(1, spatialSize**3))
                        if S['dense']:
                            data_ = data_.toarray().reshape((1, spatialSize**3))                          
                    elif postproc == 'haar':
                        data_ = np.zeros((spatialSize, spatialSize, spatialSize))
                        data_[x,y,z] = features[:,s]
                        for d in range(3):    
                            data_ = np.tensordot(haarD[spatialSize], data_, (1,d)) 
                        data_ = data_.reshape((1,-1))
                        if not S['dense']:
                            data_ = sparse.csr_matrix(data_)
                    elif postproc == 'minmax':
                        data_ = np.array([np.min(features[:,s]), np.max(features[:,s])]).reshape((1,2))                        
                        if not S['dense']:
                            data_ = sparse.csr_matrix(data_)
                    elif postproc == 'hist':
                        data_ = np.percentile(features[:,s], [0, 25, 50, 75, 100]).reshape((1,5))                        
                        if not S['dense']:
                            data_ = sparse.csr_matrix(data_)
                 
                    data.append(data_)

            fL.extend(data)
        if S['dense']:
            features = np.concatenate(fL, axis=1).reshape((1,-1))
        else:
            features = sparse.hstack(fL)
        X.append(features)
    if S['dense']:
        X = np.concatenate(X, axis=0) 
    else:
        X = sparse.vstack(X)
    return X  


def genXdataMP(dataSettings):
    ''' Generate Nrepeats batches of X data by distributing jobs to Nworker processes'''
    S = dataSettings
    DF = {}
    Xall = {}
    for label in ['train', 'test']:
        print 'Generating', S['Nrepeats'], label, 'batches:',
        DF[label] = pd.read_csv(S['pathTemplate'] %(label), header=None)   
        Xall[label] = []
        results = []
        pool = multiprocessing.Pool(S['Nworkers'])
        seedA = np.random.randint(0, 2**30, size=(S['Nrepeats'],))
        for rep in range(S['Nrepeats']):                    
            results.append(pool.apply_async(getXsingle, (rep, seedA[rep], DF, label, S
                                                         )))
        for n, res in enumerate(results):
            print n,
            Xall[label].append(res.get())
        pool.close()
        if S['dense']:
            Xall[label] = np.concatenate(Xall[label], axis=0)
        else:
            Xall[label] = sparse.vstack(Xall[label])
        print 
    return Xall          

 
def genYdata(dataSettings):
    S = dataSettings
    DF = {}
    Y = {}
    for label in ['train', 'test']:
        DF[label] = pd.read_csv(S['pathTemplate'] %(label), header=None, index_col=0)
        if S['dataset'] == 'ESB':
            DF[label]['class'] = DF[label].index.str.rsplit('/', n=1).str.get(0)
        elif S['dataset'] == 'ModelNet':
            DF[label]['class'] = DF[label].index.str.split('/', n=1).str.get(0)
        else:
            raise NotImplementedError
     
    le = LabelEncoder()
    le.fit(DF['train']['class'])
    for label in ['train', 'test']:
        DF[label]['class_num'] = le.transform(DF[label]['class'])
        Y[label] = np.array(DF[label]['class_num'])

    return Y, DF, le


def genXYdata(dataSettings):
    S = dataSettings
    if S['dataset'] == 'ESB':
        S['rootPath'] = '../datasets/ESB'
        S['pathTemplate'] = os.path.join(S['rootPath'], 'ESB_%s')
    elif S.dataset == 'ModelNet':
        S['rootPath'] = '../datasets/ModelNet40'
        S['pathTemplate'] = os.path.join(S['rootPath'], 'ModelNet_%s_files.txt')
    else:
        raise NotImplementedError    
    
    # override for non-random fitting strategy
    if S['modeFitToCube'] == 'pca':
        S['Nrepeats'] = 1
    elif S['modeFitToCube'] == 'pca_mult':
        S['Nrepeats'] = 48    
                      
    dimD = {'Bool':1,
            'SA':1,
            'AN':3,
            'QF':6,
            'EV':3,
            'VAD':1,
            'EAD':1,
            'VE':1}
    
    feature_names = []
    for feat_ in S['featureL']:
        spatialSize = feat_[0]
        for feat in feat_[1]:            
            dim = dimD[feat]
            for d in range(dim):
                for postproc in feat_[2]:
                    if postproc == 'voxel':
                        fL = [None]*(spatialSize**3)
                        for x in range(spatialSize):
                            for y in range(spatialSize):
                                for z in range(spatialSize):
                                    fL[x+spatialSize*y+spatialSize*spatialSize*z] = \
                                        str(spatialSize)+feat+str(d)+'_'+str(x)+'_'+str(y)+'_'+str(z)                                                
                    elif postproc == 'haar':
                        fL = [str(spatialSize)+feat+str(d)+'haar'+str(s) for s in range(spatialSize**3)]
                    elif postproc == 'minmax':
                        fL = [str(spatialSize)+feat+str(d)+'min', str(spatialSize)+feat+str(d)+'max']
                    elif postproc == 'hist':
                        fL = [str(spatialSize)+feat+str(d)+'hist'+ str(a) for a in [0, 25, 50, 75, 100]]
                    else:
                        raise NotImplementedError
                    feature_names.extend(fL)
    
    Y, DF, le = genYdata(S)
    Nclasses = len(set(Y['train']))
    Yrepd = {}
    for label in ['train', 'test']:
        Yrepd[label] = np.concatenate([Y[label] for _ in range(S['Nrepeats'])], axis=0) 
    X = genXdataMP(S)
        
    return X, Y, Yrepd, DF, le, Nclasses, feature_names, dataSettings


if __name__ == "__main__":
    
    # example of settings
    dataSettings = {
                'dataset': 'ESB', # ESB or ModelNet
                'featureL': [ # the features to be generated, as (spatialSize, list of feature types, list of postproc strategies) 
                             (2, ['Bool', 'SA', 'QF', 'EV', "VAD", "EAD", 'VE'], ['voxel', 'hist']),
                             (4, ['Bool', 'SA', 'QF', 'EV', "VAD", "EAD", 'VE'], ['voxel', 'haar', 'hist']),
                             (6, ['Bool', 'SA', 'QF', 'EV', "VAD", "EAD", 'VE'], ['hist']),
                             (8, ['SA', 'QF', 'EV', "VAD", "EAD", 'VE'], ['hist']),
                             (16, ['SA', 'QF', 'EV', "VAD", "EAD", 'VE'], ['hist']),
                             (32, ['SA', 'QF', 'EV', "VAD", "EAD", 'VE'], ['hist']),
                             (64, ['SA', 'QF', 'EV', "VAD", "EAD", 'VE'], ['hist']),
                             (128, ['SA', 'QF', 'EV', "VAD", "EAD", 'VE'], ['hist']),
                            ],
                'Nworkers': 5, 
                'Nrepeats': 2, # train/test will be repeated this many times, with different/random transformations 
                'dense': False, # return X as dense arrays
                'randShift': False, 
                'randRotAngle': 2*np.pi,
                'vertical': False,
                'modeFitToCube': ['pca', 'pca_mult', 'scaleShift'][2]
                }   
    
    # generate the arrays for training and testing 
    X, Y, Yrepd, DF, le, Nclasses, feature_names, dataSettings = genXYdata(dataSettings)
    
    
    
    
    
    
    
