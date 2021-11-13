import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class PCA:
    def __init__(self,logdir=os.getcwd()):
        pathDailyData = '../data/DailyReturns-RFadjusted-old.npz' 
        pathMonthlyDataUnnormalized = '../data/MonthlyDataUnnormalized.npz'
        pathMonthlyData = '../data/MonthlyData.npz'             

        self.monthlyDataUnnormalized = np.load(pathMonthlyDataUnnormalized, allow_pickle=True)['data']
        self.monthlyCaps = np.nan_to_num(self.monthlyDataUnnormalized[:,:,19])
        dailyData = np.load(pathDailyData, allow_pickle=True)
        monthlyData = np.load(pathMonthlyData, allow_pickle=True)
        self.monthlyData = monthlyData['data']
        self.dailyData = dailyData['data']
        self.dailyDates = pd.to_datetime(dailyData['date'], format='%Y%m%d')
        self.monthlyDates = pd.to_datetime(monthlyData['date'], format='%Y%m%d')
        self._logdir = logdir    
    
    def OOSRollingWindowPermnos(self,save=True, printOnConsole=True, initialOOSYear=1998,sizeWindow = 60,sizeCovarianceWindow=252,CapProportion=0.01,factorList = range(0,16)):
        
        cap_chosen_idxs = self.monthlyCaps/np.nansum(self.monthlyCaps,axis=1,keepdims=True) >= CapProportion*0.01
        mask = (~np.isnan(self.monthlyData[:,:,0])) *cap_chosen_idxs
        Rdaily = self.dailyData.copy()
        T,N = Rdaily.shape
        firstOOSDailyIdx = np.argmax(self.dailyDates.year >= initialOOSYear)
        firstOOSMonthlyIdx = np.argmax(self.monthlyDates.year >= 1998)
        
        assetsToConsider = ( np.count_nonzero(~np.isnan(self.dailyData[firstOOSDailyIdx:,:]), axis=0) >= 30 ) \
                         & ( np.sum(mask[firstOOSMonthlyIdx:,:], axis=0) >= 1 )
        Ntilde = np.sum(assetsToConsider)
        print('N',N,'Ntilde', Ntilde)
        
        #filter by cap
        if printOnConsole:
            print("Filtering by cap")
            
        for month in range(1,len(self.monthlyDates)):
            idxs_days_month = (self.dailyDates > self.monthlyDates[month-1]) & (self.dailyDates <= self.monthlyDates[month])
            Rdaily[idxs_days_month,:] *= mask[month-1,:]
        np.nan_to_num(Rdaily,copy=False)
            
        if printOnConsole:
            print("Filtered by cap!")
            print("Computing residuals")
        
        for factor in factorList:
            residualsOOS = np.zeros((T-firstOOSDailyIdx,N), dtype=float)
            residualsMatricesOOS = np.zeros((T-firstOOSDailyIdx,Ntilde,Ntilde), dtype=np.float32)
            #print( residualsOOS.shape)
            for t in range(T-firstOOSDailyIdx):                      
                idxsSelected = ~np.any(Rdaily[(t+firstOOSDailyIdx-sizeCovarianceWindow+1):(t+firstOOSDailyIdx+1),:] == 0, axis = 0).ravel()                     
                #print(idxsSelected.shape)
                       
                if factor == 0:
                    residualsOOS[t:(t+1),idxsSelected] = Rdaily[(t+firstOOSDailyIdx):(t+firstOOSDailyIdx+1),idxsSelected]
            
                else:
                    res_cov_window = Rdaily[(t+firstOOSDailyIdx-sizeCovarianceWindow):(t+firstOOSDailyIdx),idxsSelected]
                    res_mean = np.mean(res_cov_window, axis=0, keepdims=True)
                    res_vol = np.sqrt(np.mean((res_cov_window-res_mean)**2,axis=0,keepdims=True))
                    res_normalized = (res_cov_window - res_mean) / res_vol
                    Corr = np.dot(res_normalized.T, res_normalized)
                    eigenValues, eigenVectors = np.linalg.eig(Corr)
                    temp = np.argpartition(-eigenValues, factor)
                    idxs = temp[:factor]
                    loadings = eigenVectors[:,idxs].real   #takes eigenvector corresponding to factor largest eigenvalues                  
                    factors = np.dot(res_cov_window[-sizeWindow:,:]/res_vol, loadings)
                    DayFactors = np.dot(Rdaily[t+firstOOSDailyIdx,idxsSelected]/res_vol, loadings) 
                    old_loadings = loadings
                    regr = LinearRegression(fit_intercept=False,n_jobs=48).fit(factors,res_cov_window[-sizeWindow:,:])
                    loadings = regr.coef_
                    residuals = Rdaily[t+firstOOSDailyIdx,idxsSelected] - DayFactors.dot(loadings.T)
                    residualsOOS[t:(t+1),idxsSelected] = residuals
                    
                    Nprime = len(res_cov_window[-1:,:].ravel())
                    MatrixFull = np.zeros((N,N))
                    #print(res_vol.shape,np.diag(res_vol.shape).shape,old_loadings.shape,loadings.T.shape,(np.eye(Nprime)-np.diag(res_vol.squeeze())@old_loadings@loadings.T).shape)
                    MatrixReduced = (np.eye(Nprime)-np.diag(1/res_vol.squeeze())@old_loadings@loadings.T)
                    #print(np.max(factors[-1:,:].dot(loadings.T)-res_cov_window[-1:,:]@np.diag(1/res_vol.squeeze())@old_loadings@loadings.T))
                    #print(np.max(res_cov_window[-1:,:].shape,np.eye(Nprime).shape,np.eye(Nprime)))
                    #print(np.max( res_cov_window[-1:,:]-res_cov_window[-1:,:]@np.eye(Nprime)))
                    idxsSelected2 = idxsSelected.reshape((N,1))@idxsSelected.reshape((1,N))
                    #breakpoint()
                    MatrixFull[idxsSelected2] = MatrixReduced.ravel()
                    #print(np.count_nonzero(MatrixFull))
                    #print(np.max(MatrixFull[idxsSelected][:,idxsSelected].ravel()-MatrixFull[idxsSelected2]))
                    #print(np.sum(assetsToConsider*idxsSelected),np.sum(assetsToConsider),np.sum(idxsSelected))
                    residuals2 = res_cov_window[-1:,:] @ MatrixReduced
                    #print(np.max(residuals-residuals2))#, residuals, residuals2)
                    
                    #print(np.count_nonzero(MatrixFull[assetsToConsider][:,assetsToConsider]))
                    #print(np.count_nonzero(MatrixFull[assetsToConsider.reshape((N,1))@assetsToConsider.reshape(1,N)]))
                    residualsMatricesOOS[t:(t+1)] = MatrixFull[assetsToConsider][:,assetsToConsider].T 
                    #print(np.linalg.norm(residualsOOS[t:(t+1),assetsToConsider]-residualsMatricesOOS[t:(t+1)]@Rdaily[(t+firstOOSDailyIdx),assetsToConsider]))
                    #print(np.count_nonzero(MatrixReduced), np.count_nonzero(residualsMatricesOOS[t:t+1]))
                    #break
                if t%50==0 and printOnConsole:
                    print(f"At date {t}/{T-firstOOSDailyIdx}, Number of permnos: {np.sum(idxsSelected)}") 
                    print(np.linalg.norm(residualsOOS[t:(t+1),assetsToConsider]-residualsMatricesOOS[t:(t+1)]@Rdaily[(t+firstOOSDailyIdx),assetsToConsider]))
                     
                      
            np.nan_to_num(residualsOOS,copy=False)
            np.nan_to_num(residualsMatricesOOS,copy=False)
            Ttilda,Ntilda = Rdaily[firstOOSDailyIdx:,assetsToConsider].shape
            #print(np.max(residualsOOS[:,assetsToConsider] - np.matmul(residualsMatricesOOS,Rdaily[firstOOSDailyIdx:,assetsToConsider].reshape(Ttilda,Ntilda,1)).squeeze()))
            
            if printOnConsole:  
                logging.info(f"Finished! Cap: {CapProportion}, factor: {factor}")
            if save:
                if factor == 20: 
                    np.save(os.path.join(self._logdir, f"AvPCA_OOSresiduals_{factor}_factors_{initialOOSYear}_initialOOSYear_{sizeWindow}_rollingWindow_{sizeCovarianceWindow}_covWindow_{CapProportion}_Cap.npy"), residualsOOS)
                np.save(os.path.join(self._logdir, f"AvPCA_OOSMatrixresiduals_{factor}_factors_{initialOOSYear}_initialOOSYear_{sizeWindow}_rollingWindow_{sizeCovarianceWindow}_covWindow_{CapProportion}_Cap.npy"), residualsMatricesOOS)
        return    
    
    def OOSRollingWindowPermnosVectorized(self,save=True, printOnConsole=True, initialOOSYear=1998,sizeWindow = 60,sizeCovarianceWindow=252,CapProportion=0.01,factorList = range(0,16)):  
        Rdaily = self.dailyData.copy() #np.nan_to_num(self.dailyData)
        T,N = Rdaily.shape
        firstOOSDailyIdx = np.argmax(self.dailyDates.year >= initialOOSYear)
        firstOOSMonthlyIdx = np.argmax(self.monthlyDates.year >= initialOOSYear)
        OOSDailyDates = self.dailyDates[firstOOSDailyIdx:] 
        cap_chosen_idxs = self.monthlyCaps/np.nansum(self.monthlyCaps,axis=1,keepdims=True) >= CapProportion*0.01
        mask = (~np.isnan(self.monthlyData[:,:,0])) *cap_chosen_idxs
                 
        if printOnConsole:
            print("Computing residuals")
        
        residualsOOS = np.zeros((len(factorList),T-firstOOSDailyIdx,N), dtype=float)
        monthlyIdx = firstOOSMonthlyIdx-2
        for t in range(T-firstOOSDailyIdx):    
            if self.dailyDates[t+firstOOSDailyIdx-1].month != self.dailyDates[t+firstOOSDailyIdx].month:
                monthlyIdx +=1 
            idxsNotMissingValues = ~np.any(np.isnan(Rdaily[(t+firstOOSDailyIdx-sizeCovarianceWindow+1):(t+firstOOSDailyIdx+1),:]), axis = 0).ravel()                     
            idxsSelected = idxsNotMissingValues * mask[monthlyIdx,:]
            if t%50==0 and printOnConsole:                    
                print(f"At date {t}/{T-firstOOSDailyIdx}, Not-missing permnos: {np.sum(idxsNotMissingValues)}, Permnos with cap {np.sum(mask[monthlyIdx,:])}, Selected: {sum(idxsSelected)}")                              
            res_cov_window = Rdaily[(t+firstOOSDailyIdx-sizeCovarianceWindow+1):(t+firstOOSDailyIdx+1),idxsSelected]
            res_mean = np.mean(res_cov_window,axis=0,keepdims=True)
            res_vol = np.sqrt(np.mean((res_cov_window-res_mean)**2,axis=0,keepdims=True))
            res_normalized = (res_cov_window - res_mean) / res_vol
            Corr = np.dot(res_normalized.T, res_normalized)
            #eigenValues, eigenVectors = np.linalg.eig(Corr)
            eigenValues, eigenVectors = np.linalg.eigh(Corr)
            for (i,factor) in enumerate(factorList):
                if factor == 0:
                    residualsOOS[i:(i+1),t:(t+1),idxsSelected] = Rdaily[(t+firstOOSDailyIdx):(t+firstOOSDailyIdx+1),idxsSelected]
                    #print(residualsOOS[i:(i+1),t:(t+1),idxsSelected].shape,Rdaily[(t+firstOOSDailyIdx):(t+firstOOSDailyIdx+1),idxsSelected].shape)
                else:                
                    #temp = np.argpartition(-eigenValues, factor)
                    #idxs = temp[:factor]
                    #loadings = eigenVectors[:,idxs].real   #takes eigenvector corresponding to factor largest eigenvalues                  
                    loadings = eigenVectors[:,-factor:].real
                    factors = np.dot(res_cov_window[-sizeWindow:,:]/res_vol, loadings) 
                    old_loadings = loadings
                    regr = LinearRegression(fit_intercept=False,n_jobs=-1).fit(factors,res_cov_window[-sizeWindow:,:])
                    loadings = regr.coef_
                    residuals = res_cov_window[-1:,:] - factors[-1:,:].dot(loadings.T)
                    residualsOOS[i:(i+1),t:(t+1),idxsSelected] = residuals

        np.nan_to_num(residualsOOS,copy=False)
        
        if printOnConsole:  
            logging.info(f"Finished! Cap: {CapProportion}")
        if save:
            for (i,factor) in enumerate(factorList):
                np.save(os.path.join(self._logdir, f"AvPCA_OOSresiduals_{factor}_factors_{initialOOSYear}_initialOOSYear_{sizeWindow}_rollingWindow_{sizeCovarianceWindow}_covWindow_{CapProportion}_Cap.npy"), residualsOOS[i,:,:])
        return        
