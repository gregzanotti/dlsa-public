import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from utils import initialize_logging, nploadp


class FamaFrench:
    def __init__(self, logdir=os.getcwd()):
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
        self.FamaFrenchFiveFactorsDaily = pd.read_csv("../data/FamaFrench8Daily.csv", index_col=0)/100
        print(self.FamaFrenchFiveFactorsDaily.head())
        #breakpoint()
    
    
    def OOSRollingWindowPermnos(self, save=True, printOnConsole=True, initialOOSYear=1998, sizeWindow=60, cap=0.01, listFactors=list(range(8))):
        Rdaily = self.dailyData.copy() #np.nan_to_num(self.dailyData)
        T,N = Rdaily.shape
        firstOOSDailyIdx = np.argmax(self.dailyDates.year >= initialOOSYear)
        firstOOSMonthlyIdx = np.argmax(self.monthlyDates.year >= initialOOSYear)
        firstOOSFFDailyIdx = np.argmax(self.FamaFrenchFiveFactorsDaily.index >= initialOOSYear*10000)
        FamaFrenchDaily = self.FamaFrenchFiveFactorsDaily.to_numpy()
        OOSDailyDates = self.dailyDates[firstOOSDailyIdx:] 
        cap_chosen_idxs = self.monthlyCaps/np.nansum(self.monthlyCaps,axis=1,keepdims=True) >= cap*0.01
        mask = (~np.isnan(self.monthlyData[:,:,0])) *cap_chosen_idxs
        
        filename = f"DailyFamaFrench_OOSresiduals_{3}_factors_{initialOOSYear}_initialOOSYear_{sizeWindow}_rollingWindow_{cap}_Cap.npy"
        DataTrain = np.load(os.path.join(self._logdir, filename))
        #chooses stocks which have at least #lookback non-missing observations in all the training time
        assetsToConsider = np.count_nonzero(DataTrain,axis=0)>=30
        Ntilde = np.sum(assetsToConsider)
        print('N',N,'Ntilde', Ntilde)
          
        if printOnConsole:
            print("Computing residuals")
        
        for factor in listFactors:
            residualsOOS = np.zeros((T-firstOOSDailyIdx,N), dtype=float)
            notmissingOOS = np.zeros((T-firstOOSDailyIdx), dtype=float)
            monthlyIdx = firstOOSMonthlyIdx-2
            residualsMatricesOOS = np.zeros((T-firstOOSDailyIdx,Ntilde,Ntilde+factor), dtype=np.float32)
            
            for t in range(T-firstOOSDailyIdx): 
                if self.dailyDates[t+firstOOSDailyIdx-1].month != self.dailyDates[t+firstOOSDailyIdx].month:
                    monthlyIdx +=1 
                idxsNotMissingValues = ~np.any(np.isnan(Rdaily[(t+firstOOSDailyIdx-sizeWindow):(t+firstOOSDailyIdx),:]), axis = 0).ravel()                     
                #print(idxsNotMissingValues.shape,mask[monthlyIdx,:].shape)
                #print(self.monthlyDates[monthlyIdx],OOSDailyDates[t])
                idxsSelected = idxsNotMissingValues * mask[monthlyIdx,:]
                notmissingOOS[t]=np.sum(idxsNotMissingValues)
                
                if t%100==0 and printOnConsole:                    
                    print(f"At date {OOSDailyDates[t]}, Not-missing permnos: {notmissingOOS[t]}, "
                          f"Permnos with cap {np.sum(mask[monthlyIdx,:])}, Selected: {sum(idxsSelected)}")                              
                    print(np.sum(idxsSelected)-np.sum(assetsToConsider*idxsSelected))
                if factor == 0:
                    residualsOOS[t:(t+1),idxsSelected] = Rdaily[(t+firstOOSDailyIdx):(t+firstOOSDailyIdx+1),idxsSelected]          
                    residualsMatricesOOS[t:(t+1),:,:Ntilde] = np.diag(idxsSelected[assetsToConsider])
                else:
                    Y =  Rdaily[(t+firstOOSDailyIdx-sizeWindow):(t+firstOOSDailyIdx),idxsSelected] 
                    X = FamaFrenchDaily[(t+firstOOSFFDailyIdx-sizeWindow):(t+firstOOSFFDailyIdx),:factor]
                    regr = LinearRegression(fit_intercept=False,n_jobs=48).fit(X,Y)
                    loadings = regr.coef_.T #5 x N
                    OOSreturns = Rdaily[(t+firstOOSDailyIdx):(t+firstOOSDailyIdx+1),idxsSelected]
                    factors =  FamaFrenchDaily[(t+firstOOSFFDailyIdx):(t+firstOOSFFDailyIdx+1),:factor]#TxnFactors
                    residuals = OOSreturns - factors.dot(loadings)
                    residualsOOS[t:(t+1),idxsSelected] = np.nan_to_num(residuals,copy=False)
                    
                    Loadings = np.zeros((N,factor))
                    Loadings[idxsSelected] = -loadings.T
                    residualsMatricesOOS[t,:,:Ntilde] = np.diag(idxsSelected[assetsToConsider])#np.eye(Ntilde)
                    residualsMatricesOOS[t,:,Ntilde:] = np.nan_to_num(Loadings[assetsToConsider], copy=False)
                    if t%50==0 and printOnConsole:
                        concatenate = np.concatenate((np.nan_to_num(Rdaily[(t+firstOOSDailyIdx),assetsToConsider],copy=False),
                                                     FamaFrenchDaily[(t+firstOOSFFDailyIdx),:factor]),axis=0)
                        print(np.linalg.norm(residualsOOS[t,assetsToConsider]-residualsMatricesOOS[t]@concatenate))
            
            print("Transforming NaNs to nums")
            np.nan_to_num(residualsOOS,copy=False)
            np.nan_to_num(residualsMatricesOOS,copy=False)
            if printOnConsole:
                logging.info(f"Finished! Cap: {cap}, factor: {factor}")
            if save:
                logging.info(f"Saving")
                residuals_mtx_filename = f"DailyFamaFrench_OOSMatrixresiduals" + \
                                         f"_{factor}_factors_{initialOOSYear}_initialOOSYear_{sizeWindow}_rollingWindow_{cap}_Cap.npy"
                np.save(os.path.join(self._logdir, residuals_mtx_filename), residualsMatricesOOS)
                residuals_filename = f"DailyFamaFrench_OOSresiduals" + \
                                     f"_{factor}_factors_{initialOOSYear}_initialOOSYear_{sizeWindow}_rollingWindow_{cap}_Cap.npy"
                np.save(os.path.join(self._logdir, residuals_filename), residualsOOS)
                logging.info(f"Saved")

