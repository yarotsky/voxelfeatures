'''
Construst and test an XGBoost-based model for the ESB dataset. 

@author: dmitry.yarotsky
'''
import os
import numpy as np
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xgboost as xgb
import cPickle

import voxelfeatures as vf 
import preprocXY


def trainTestModel(dataSettings, modelSettings):    
    X, Y, Yrepd, DF, le, Nclasses, _, dataSettings = preprocXY.genXYdata(dataSettings)
              
    Nrepeats = dataSettings['Nrepeats']

    dtrain = xgb.DMatrix(X['train'], label=Yrepd['train'])
    dtest = xgb.DMatrix(X['test'], label=Yrepd['test'])
        
    xgb_param = {'num_class': Nclasses, 'silent': 1,  'objective':'multi:softprob'}
    for key in modelSettings.keys():
        if key != 'num_round':
            xgb_param[key] = modelSettings[key]
            
    xgb_num_round = modelSettings['num_round']
    evallist  = [(dtest,'eval'), (dtrain,'train')]   
    
    print '=== training the model'          
    bst = xgb.train(xgb_param, dtrain, xgb_num_round, evallist)  

    YpredTest_proba = np.array(bst.predict(dtest)) 
    YpredTestProbaMean = YpredTest_proba.reshape((Nrepeats, len(Y['test']), -1)).mean(axis=0)
            
    YpredTest = np.argmax(YpredTestProbaMean, 1)      
    testError = np.sum(YpredTest != Y['test']).astype('float')/len(YpredTest)   
    print 'Test error (share of wrong predictions):', testError 
    
    return bst, DF, Y, le, YpredTestProbaMean, YpredTest, dataSettings 


def plotPredictions(bst, DF, Y, le, YpredTestProbaMean, YpredTest, dataSettings):   
    testError = np.sum(YpredTest != Y['test']).astype('float')/len(YpredTest)   
    print 'Test error (share of wrong predictions):', testError 
    print '======== Examples of predictions'
    
    outer_grid = gridspec.GridSpec(3, 3, wspace=0.1, hspace=0.2)
    randTestSubset = np.random.permutation(len(Y['test']))[:9]
    
    bgcolor = (0.8, 0.8, 0.8)
    fig = plt.figure(figsize=(9,6), facecolor=bgcolor)
    for m in range(9):
        print '--------', m
        m1 = randTestSubset[m]
        print 'object:', DF['test'].index[m1]
        print 'most probable classes:'
        order1 = np.argsort(YpredTestProbaMean[m1])[::-1]
        for n in range(4):
            n1 = order1[n]
            print '---', YpredTestProbaMean[m1][n1], le.inverse_transform(n1) 
        print 'predicted class:', le.inverse_transform(YpredTest[m1])
        print 'predicted class probability:', np.max(YpredTestProbaMean[m1])
        print 'true class:', le.inverse_transform(Y['test'][m1])
        print 'true class probability:', YpredTestProbaMean[m1, Y['test'][m1]] 
        
        mlab.figure(bgcolor=bgcolor, 
                    size=(350,400))
        vertA, faceA = vf.getDataOff(os.path.join(dataSettings['rootPath'], 
                                                  DF['test'].index[m1]))
        
        mlab.triangular_mesh(vertA[:,0], vertA[:,1], vertA[:,2], faceA) 
        screenshot = mlab.screenshot()   
        
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 4,
            subplot_spec = outer_grid[m], wspace=0.0, hspace=0.0)   
        
        ax = plt.subplot(inner_grid[0:2])
        plt.imshow(screenshot)
        plt.axis('off')
        
        ax = plt.subplot(inner_grid[2:4])
        plt.xlim([0,1])
        pos = np.arange(5)+0.5
        classNames = [le.inverse_transform(order1[n]) for n in range(5)][::-1]
        classProbs = [YpredTestProbaMean[m1][order1[n]] for n in range(5)][::-1]
        rects = ax.barh(pos, classProbs, align='center', height=0.95, color=(1,0.5,1)) 
        ax.set_yticks([])
        for n, rect in enumerate(rects):
            yloc = rect.get_y()+rect.get_height()/2.0
            xlim = ax.get_xlim()
            ax.text(0.02*(xlim[1]-xlim[0]), yloc, classNames[n].replace('/','/\n'), horizontalalignment='left',
             verticalalignment='center', color='k', fontsize=7)
            if le.inverse_transform(Y['test'][m1]) == classNames[n]: # true class
                rect.set(color=(0.5,1,0.5))
            
        max_xticks = 3
        xloc = plt.MaxNLocator(max_xticks)
        ax.xaxis.set_major_locator(xloc)

    fig.subplots_adjust(left=0.02, bottom=0.05, right=0.98, top=0.98)   
    plt.show()
    
    
def main(retrain=False, saveResults=True):
    dataSettings = {
            'dataset': 'ESB',
            'featureL': [
                         (2, ['Bool', 'SA', 'QF', 'EV', "VAD", "EAD", 'VE'], ['voxel', 'hist']),
                         (4, ['Bool', 'SA', 'QF', 'EV', "VAD", "EAD", 'VE'], ['voxel', 'haar', 'hist']),
                         (8, ['SA', 'QF', 'EV', "VAD", "EAD", 'VE'], ['hist']),
                         (16, ['SA', 'QF', 'EV', "VAD", "EAD", 'VE'], ['hist']),
                         (32, ['SA', 'QF', 'EV', "VAD", "EAD", 'VE'], ['hist']),
                         (64, ['SA', 'QF', 'EV', "VAD", "EAD", 'VE'], ['hist']),
                         (128, ['SA', 'QF', 'EV', "VAD", "EAD", 'VE'], ['hist']),
                        ],
            'Nworkers': 5,
            'Nrepeats': 20,
            'dense': False,
            'randShift': False,
            'randRotAngle': 2*np.pi,
            'vertical': False,
            'modeFitToCube': 'scaleShift'
            }
    
    modelSettings = {'max_depth': 2,
                     'num_round': 100,
                     'subsample': 0.5,
                     'colsample_bytree': 0.5,
                     'colsample_bylevel': 0.5,
                    }
    if retrain:
        bst, DF, Y, le, YpredTestProbaMean, YpredTest, dataSettings = trainTestModel(dataSettings, modelSettings)
        
        if saveResults:
            bst.save_model('model.bst')
            with open('results.pkl', 'w') as f:
                cPickle.dump([DF, Y, le, YpredTestProbaMean, YpredTest, dataSettings], f)
        
    if not retrain:
        bst = xgb.Booster()
        bst.load_model('model.bst')
        with open('results.pkl', 'r') as f:
            DF, Y, le, YpredTestProbaMean, YpredTest, dataSettings = cPickle.load(f)        
                
    plotPredictions(bst, DF, Y, le, YpredTestProbaMean, YpredTest, dataSettings)
        
        
if __name__ == "__main__":
    main()    
        
   
    
                                    
    
          


   

