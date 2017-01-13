'''
Various experiments with XGBoost-based models on the ESB dataset. 

@author: dmitry.yarotsky
'''

import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import scipy.sparse as sparse
import pandas as pd
import re

import preprocXY

    
def singleFeatures():  
    ''' Performance of individual features. ''' 
    def trainTestSave():        
        modelSettings = {
                         'num_round': 100,
                         'subsample': 0.5,
                        }
        
        for n in range(0,6):
            resol = int(2**n)
            print '+++++ resolution', resol
        
            dataSettings = {
                    'dataset': 'ESB',
                    'featureL': [
                                 (resol, ['Bool', 'SA', 'AN', 'QF', 'EV', "VAD", "EAD", 'VE'], ['voxel'])
                                ],
                    'Nworkers': 5,
                    'Nrepeats': 20,
                    'dense': False,
                    'randShift': False,
                    'randRotAngle': 2*np.pi,
                    'vertical': False,
                    'modeFitToCube': 'scaleShift'
                    }
              
            X, Y, Yrepd, _, _, Nclasses, feature_names, _ = preprocXY.genXYdata(dataSettings)
            
            X['train'] = sparse.csc_matrix(X['train'])
            X['test'] = sparse.csc_matrix(X['test'])
                    
            featToUse = []
            resD = {}
            featKeyL = ['Bool', 'SA', 'AN', 'QF', 'EV', "VAD", "EAD", 'VE']
            for featKey in featKeyL:
                print '=== ', featKey
                featToUse = [k for k in range(X['train'].shape[1]) if featKey in feature_names[k]]
                
                print 'Used features:', len(featToUse)
                        
                dtrain = xgb.DMatrix(X['train'][:,featToUse], label=Yrepd['train'])
                dtest = xgb.DMatrix(X['test'][:,featToUse], label=Yrepd['test'])
                    
                xgb_param = {'num_class': Nclasses, 'silent': 1,  'objective':'multi:softprob'}
                for key in modelSettings.keys():
                    if key != 'num_round':
                        xgb_param[key] = modelSettings[key]
                        
                xgb_num_round = modelSettings['num_round']
                evallist  = [(dtest,'eval'), (dtrain,'train')]               
                
                bst = xgb.train(xgb_param, dtrain, xgb_num_round, evallist)  
            
                YpredTest_proba = np.array(bst.predict(dtest)) 
                YpredTestProbaMean = YpredTest_proba.reshape((dataSettings['Nrepeats'], len(Y['test']), -1)).mean(axis=0)
                        
                YpredTest = np.argmax(YpredTestProbaMean, 1)      
                testError = np.sum(YpredTest != Y['test']).astype('float')/len(YpredTest)  
                resD[featKey] = testError
                print 'Test error (share of wrong predictions):', testError                     
                
            resDF = pd.DataFrame(index=featKeyL, data=[resD[key] for key in featKeyL], columns=['error'])
            resDF.to_csv('errors%s.csv' %(resol), index_label='feature')
                      
    def plot():
        errDF = []
        for n in range(6):
            resol = int(2**n)
            df = pd.read_csv('errors%s.csv' %(resol), index_col='feature')
            df.columns = [str(resol)]
            errDF.append(df)  
        errDF = pd.concat(errDF, axis=1)
        print errDF 
        
        
        lines = [
                {'width':4, 'style':'-', 'color':(0.7,0.7,0.7)}, 
                {'width':4.5, 'style':':', 'color':(0.6,0.6,0)},
                {'width':3, 'style':':', 'color':(0.2,0,0.2)},
                {'width':1.5, 'style':'-', 'color':(0,0,0.4)},
                {'width':3, 'style':'--', 'color':(0,1,1)},
                {'width':2, 'style':'--', 'color':(0.4,0,0)},
                {'width':4, 'style':'-.', 'color':(1,0,0)},
                {'width':2, 'style':'-.', 'color':(0,0.2,0)},
                ]
        plt.figure(figsize=(5,5))
        for n, key in enumerate(errDF.index):
            plt.plot(range(6), errDF.loc[key], lines[n]['style'], color=lines[n]['color'], 
                linewidth=lines[n]['width'], label=key)
        ax = plt.gca()
        ax.set_xticks(range(6))
        ax.set_xticklabels(errDF.columns)
        legend = plt.legend(bbox_to_anchor=(0.8, 1))
        plt.xlabel('Resolution')
        plt.ylabel('Classification error')
        plt.tight_layout()
        plt.savefig('resol_vs_error.png')
        plt.savefig('resol_vs_error.pdf')
        plt.show() 
    
    trainTestSave()
    plot()
    

def rawVsHaar():   
    ''' Raw vs Haar-transformed features. ''' 
    def trainTestSave():        
        modelSettings = {
                         'num_round': 100,
                         'subsample': 0.5,
                        }
        
        for n in range(0,5):
            resol = int(2**n)
            print '+++++ resolution', resol
        
            dataSettings = {
                    'dataset': 'ESB',
                    'featureL': [
                                 (resol, ['EV'], ['voxel', 'haar'])
                                ],
                    'Nworkers': 5,
                    'Nrepeats': 20,
                    'dense': False,
                    'randShift': False,
                    'randRotAngle': 2*np.pi,
                    'vertical': False,
                    'modeFitToCube': 'scaleShift'
                    }
              
            X, Y, Yrepd, _, _, Nclasses, feature_names, _ = preprocXY.genXYdata(dataSettings)
            
            X['train'] = sparse.csc_matrix(X['train'])
            X['test'] = sparse.csc_matrix(X['test'])
                    
            featToUse = []
            resD = {}
            featKeyL = ['raw','haar']
            for featKey in featKeyL:
                print '=== ', featKey
                if featKey == 'haar':
                    featToUse = [k for k in range(X['train'].shape[1]) if 'haar' in feature_names[k]]
                else:
                    featToUse = [k for k in range(X['train'].shape[1]) if not 'haar' in feature_names[k]]
                
                print 'Number of used features:', len(featToUse)
                        
                dtrain = xgb.DMatrix(X['train'][:,featToUse], label=Yrepd['train'])
                dtest = xgb.DMatrix(X['test'][:,featToUse], label=Yrepd['test'])
                    
                xgb_param = {'num_class': Nclasses, 'silent': 1,  'objective':'multi:softprob'}
                for key in modelSettings.keys():
                    if key != 'num_round':
                        xgb_param[key] = modelSettings[key]
                        
                xgb_num_round = modelSettings['num_round']
                evallist  = [(dtest,'eval'), (dtrain,'train')]               
                
                bst = xgb.train(xgb_param, dtrain, xgb_num_round, evallist)  
            
                YpredTest_proba = np.array(bst.predict(dtest)) 
                YpredTestProbaMean = YpredTest_proba.reshape((dataSettings['Nrepeats'], len(Y['test']), -1)).mean(axis=0)
                        
                YpredTest = np.argmax(YpredTestProbaMean, 1)      
                testError = np.sum(YpredTest != Y['test']).astype('float')/len(YpredTest)  
                resD[featKey] = testError
                print 'Test error (share of wrong predictions):', testError                     
                
            resDF = pd.DataFrame(index=featKeyL, data=[resD[key] for key in featKeyL], columns=['error'])
            resDF.to_csv('errorsRawVsHaar%s.csv' %(resol), index_label='feature')
                      
    def plot():
        errDF = []
        for n in range(5):
            resol = int(2**n)
            df = pd.read_csv('errorsRawVsHaar%s.csv' %(resol), index_col='feature')
            df.columns = [str(resol)]
            errDF.append(df)  
        errDF = pd.concat(errDF, axis=1)
        print errDF 
        
        
        lines = [
                {'width':2.5, 'style':'-', 'color':(1,0,0)}, 
                {'width':4, 'style':'--', 'color':(0,0,1)},
                ]
        plt.figure(figsize=(5,3))
        for n, key in enumerate(errDF.index):
            plt.plot(range(5), errDF.loc[key], lines[n]['style'], color=lines[n]['color'], 
                linewidth=lines[n]['width'], label=key[0].upper()+key[1:]+' EV')
        ax = plt.gca()
        ax.set_xticks(range(5))
        ax.set_xticklabels(errDF.columns)
        plt.legend()
        plt.xlabel('Resolution')
        plt.ylabel('Classification error')
        plt.tight_layout()
        plt.savefig('raw_vs_haar.png')
        plt.savefig('raw_vs_haar.pdf')
        plt.show() 
    
    trainTestSave()
    plot()

       
def errorConverge():
    ''' Full-fledged feature set, error convergence plot and feture importances histograms. ''' 
    def trainTestSave(retrain=True, saveResults=True):        
        modelSettings = {
                         'max_depth': 2,
                         'num_round': 100,
                        'subsample': 0.5,
                        'colsample_bytree': 0.5,
                        'colsample_bylevel': 0.5,
                        }                
        dataSettings = {
                'dataset': 'ESB',
                'featureL': [
                            (1, ['SA', 'AN', 'QF', 'EV', "VAD", "EAD", 'VE'], ['voxel']),
                            (2, ['SA', 'AN', 'QF', 'EV', "VAD", "EAD", 'VE'], ['voxel', 'hist']),
                            (4, ['SA', 'AN', 'QF', 'EV', "VAD", "EAD", 'VE'], ['voxel', 'hist']),
                            (8, ['SA', 'AN', 'QF', 'EV', "VAD", "EAD", 'VE'], ['hist']),
                            (16, ['SA', 'AN', 'QF', 'EV', "VAD", "EAD", 'VE'], ['hist']),
                            (32, ['SA', 'AN', 'QF', 'EV', "VAD", "EAD", 'VE'], ['hist']),
                            (64, ['SA', 'AN', 'QF', 'EV', "VAD", "EAD", 'VE'], ['hist']),
                            (128, ['SA', 'AN', 'QF', 'EV', "VAD", "EAD", 'VE'], ['hist']),
                            ],
                'Nworkers': 5,
                'Nrepeats': 20,
                'dense': False,
                'randShift': False,
                'randRotAngle': 2*np.pi,
                'vertical': False,
                'modeFitToCube': 'scaleShift'
                }
              
        X, Y, Yrepd, _, _, Nclasses, feature_names, _ = preprocXY.genXYdata(dataSettings)
        
        Nrepeats = dataSettings['Nrepeats']

        dtrain = xgb.DMatrix(X['train'], label=Yrepd['train'], feature_names=feature_names)
        dtest = xgb.DMatrix(X['test'], label=Yrepd['test'], feature_names=feature_names)
        
        xgb_param = {'num_class': Nclasses, 'silent': 1,  'objective':'multi:softprob'}
        for key in modelSettings.keys():
            if key != 'num_round':
                xgb_param[key] = modelSettings[key]
            
        evallist  = [(dtest,'eval'), (dtrain,'train')] 
                
        trainErrorL = []
        testErrorL = []
        testSymErrorL = []
        
        for n in range(modelSettings['num_round']):
            print 'Iteration', n
            if n == 0:
                bst = xgb.train(xgb_param, dtrain, 1, evallist)  
            else:
                bst = xgb.train(xgb_param, dtrain, 1, evallist, xgb_model=bst)  
    
            YpredTest_proba = np.array(bst.predict(dtest))
            YpredTest = np.argmax(YpredTest_proba, 1)      
            testError = np.sum(YpredTest != Yrepd['test']).astype('float')/len(YpredTest)   
            print 'Test error (share of wrong predictions):', testError   
            testErrorL.append(testError)           
            
            YpredTestProbaMean = YpredTest_proba.reshape((Nrepeats, len(Y['test']), -1)).mean(axis=0)   
            YpredSymTest = np.argmax(YpredTestProbaMean, 1)      
            testSymError = np.sum(YpredSymTest != Y['test']).astype('float')/len(YpredSymTest)   
            print 'Test sym error (share of wrong predictions):', testSymError  
            testSymErrorL.append(testSymError)
                        
            YpredTrain_proba = np.array(bst.predict(dtrain))
            YpredTrain = np.argmax(YpredTrain_proba, 1)      
            trainError = np.sum(YpredTrain != Yrepd['train']).astype('float')/len(YpredTrain)   
            print 'Train error (share of wrong predictions):', trainError  
            trainErrorL.append(trainError)  
            
        fscores = bst.get_fscore()
        fscores = pd.DataFrame(fscores.items(), columns=['feature', 'value'])
        fscores = fscores.sort_values('value', ascending=False)            
        fscores.to_csv('feature_importances.csv')    
        
        errDF = pd.DataFrame(data={'train': trainErrorL, 'test': testErrorL, 'testSym': testSymErrorL})
        errDF.to_csv('errHistory.csv')
        
        
    def plotHistory():
        errDF = pd.read_csv('errHistory.csv', index_col=0)
        
        plt.figure(figsize=(5,4))
        plt.plot(errDF['train'][:80], ':g', lw=4, label='train')
        plt.plot(errDF['test'][:80], '--b', lw=3, label='test')
        plt.plot(errDF['testSym'][:80], '-r', lw=2, label='test symmetrized')
        
        plt.legend(loc="upper right")
        plt.xlabel('Iteration')
        plt.ylabel('Classification error')
        plt.tight_layout()
        plt.savefig('errHistory.png')
        plt.savefig('errHistory.pdf') 
        plt.show()   
        
    def plotImportances():
        impDF = pd.read_csv('feature_importances.csv', index_col=1)
        def convertFeatureName(inp):
            m = re.match(r"(\d+)([A-Z]+)(\d)(.+)", inp)
            resol = m.group(1)
            feat = m.group(2)
            comp = m.group(3)
            if m.group(4).startswith('hist'):
                pos = m.group(4)
            else:
                assert m.group(4).startswith('_')
                pos = ','.join(m.group(4)[1:].split('_'))
            
            if resol == '1':
                if feat in ['AN', 'QF', 'EV']:
                    output = '[%s][%s][%s]'%(resol, feat, comp) 
                else:
                    output = '[%s][%s]'%(resol, feat)  
            else:                              
                if feat in ['AN', 'QF', 'EV']:
                    output = '[%s][%s][%s][%s]'%(resol, feat, pos, comp) 
                else:
                    output = '[%s][%s][%s]'%(resol, feat, pos) 
            return output
        
        fL = [convertFeatureName(f) for f in impDF.index]
                
        N = 20
        plt.figure(figsize=(5,4))
        ax = plt.gca()
        plt.barh(range(N), impDF.iloc[:N,1][::-1])
        width=0.8
        yax = ax.get_yaxis()        
        ax.set_yticks(np.arange(N) + width/2)
        ax.set_yticklabels(fL[:N][::-1], ha = 'left')

        yax.set_tick_params(pad=130)        
        plt.xlabel('Weight')
        plt.tight_layout()
        
        plt.savefig('feature_importances.png')
        plt.savefig('feature_importances.pdf')
        
        plt.show() 
        
        
    trainTestSave() 
    plotHistory()
    plotImportances()
        

       
if __name__ == "__main__":        
    #singleFeatures()
    #rawVsHaar()
    errorConverge()
        
   
    
                                    
    
          


   

