'''
Convert a dataset from .STL to .OFF.

@author: dmitry.yarotsky
'''
import locale
import warnings
import os
import subprocess
import numpy as np


if locale.localeconv()['decimal_point'] != '.':
    warnings.warn('Locale decimal separator is not "." and may cause errors when reading OFF/STL files; change LC_NUMERIC!')


def linecount(filename): # count lines in file
    with open(filename) as f:
        lineL = f.readlines()
    return len(lineL)


def getDataOff(modelpath):
    Nlines = linecount(modelpath)
    with open(modelpath,'r') as f:
        head = f.readlines(2)
    assert head[0].startswith('OFF')
    if not len(head[0].strip()) == 3: # for some reason, some ModelNet files have no line break after first line 
        Nheadlines = 1
        sizes = [int(n) for n in head[0][3:].strip().split()]
    else:
        Nheadlines = 2
        sizes = [int(n) for n in head[1].strip().split()]
    assert len(sizes) == 3 and sizes[-1] == 0
    vertA = np.genfromtxt(modelpath, dtype='float64', delimiter=' ', skip_header=Nheadlines, skip_footer=sizes[1])
    faceA = np.genfromtxt(modelpath, dtype='int64', delimiter=' ', skip_header=Nheadlines+sizes[0])
    assert len(vertA) == sizes[0]
    assert len(faceA) == sizes[1]
    assert faceA.shape[1] == 4
    assert Nlines == Nheadlines+sizes[0]+sizes[1]   
    return vertA, faceA[:,1:]     


def convertRecursivelySTL2OFF(srcFolder=None, 
                              destFolder=None,
                              method='parseXML',
                              path2meshlabMLXfilterScript=None, 
                              removeIdenticalPoints=True, 
                              verbose=True,
                              validateOFF=True):
    '''
    Recursively iterate over all .stl files in given folder and its subfolders; 
    convert each .stl file to .off file. 
    
    Two methods:
        'meshlab': calls Meshlab; requires Meshlab and the MLX filter script to merge repeated vertices. 
        'parseXML': hand-made conversion, no dependencies
    '''
    if srcFolder is None:
        srcFolder = os.getcwd()
    if destFolder is None:
        destFolder = srcFolder  #  .off files will be written beside .stl files
             
    if method == 'meshlab':           
        if path2meshlabMLXfilterScript is None:
            path2meshlabMLXfilterScript = os.path.join(os.getcwd(), 'merge_close_vertices.mlx')
        
        assert os.path.isfile(path2meshlabMLXfilterScript), 'Meshlab MLX filter script not found!'        
            
    for triple in os.walk(srcFolder):           
        for fname in triple[2]: 
            if fname.endswith('.stl') or fname.endswith('.STL'):
                if verbose:
                    print '====', fname
                fullnameSTL = os.path.join(triple[0], fname)
                destDir_ = os.path.join(destFolder, os.path.relpath(triple[0], srcFolder))
                if not os.path.exists(destDir_):
                    os.makedirs(destDir_)
                fullnameOFF = os.path.join(destDir_, fname)[:-4]+'.off'
                
                if method == 'meshlab':                
                    subprocess.call('meshlabserver -i "'+fullnameSTL+'" -o "'+fullnameOFF+
                                        '" -s "'+path2meshlabMLXfilterScript+'"', shell=True)  
                
                elif method == 'parseXML':
                    with open(fullnameSTL, 'r') as f:
                        lineL = f.readlines()
                    assert len(lineL)%7 == 2
                    Nfaces = len(lineL)/7
                    Nverts0 = 3*Nfaces
                    VertA = np.zeros((Nverts0,3))
                    lineInd = []
                    for n, line in enumerate(lineL):
                        m = n/7
                        k = n%7-3
                        if k >= 0 and k < 3:
                            lineSplit = line.strip().split()[1:]
                            assert len(lineSplit) == 3
                            for s in range(3):
                                VertA[m*3+k,s] = float(lineSplit[s])
                            lineInd.append(n)
                    b = np.ascontiguousarray(VertA).view(np.dtype((np.void, VertA.dtype.itemsize * VertA.shape[1])))
                    _, idx, idx1 = np.unique(b, return_index=True, return_inverse=True) 
                    VertAunique = VertA[idx]
                    assert len(VertA) == len(idx1) and len(VertAunique) == len(idx) 
                    data2write = ['OFF', None]
                    for n in idx:
                        data2write.append(lineL[lineInd[n]].strip().split(' ', 1)[1])
                    N0 = len(data2write) 
                    for s in range(Nfaces):  
                        if not removeIdenticalPoints or len(set(idx1[3*s:3*(s+1)])) == 3: 
                            data2write.append(' '.join(['3', str(idx1[3*s]), str(idx1[3*s+1]), str(idx1[3*s+2])]))
                    Nfaces1 = len(data2write)-N0 
                    data2write[1] = ' '.join([str(len(VertAunique)), str(Nfaces1), '0'])
                    
                    with open(fullnameOFF, 'w') as f:
                        f.write('\n'.join(data2write))
  
                    if verbose:
                        if Nfaces1 != Nfaces:
                            print 'Removed %d faces with repeated vertices' %(Nfaces-Nfaces1) 
                
                else:
                    raise NotImplementedError
                
                if validateOFF:
                    getDataOff(fullnameOFF)                
    

if __name__ == '__main__':
    convertRecursivelySTL2OFF()