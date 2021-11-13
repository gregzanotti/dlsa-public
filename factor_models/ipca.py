import os
import sys
import logging
from utils import initialize_logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def get_sharpe_tangencyPortfolio(returns): #returns has shape TxN
    if returns.shape[1] == 1:
        return np.abs(np.mean(returns))/np.std(returns)
    else:
        mean_ret = np.mean(returns, axis=0,keepdims=True)
        cov_ret = np.cov(returns.T)
        return float(np.sqrt(mean_ret @ np.linalg.solve(cov_ret,mean_ret.T)))


def get_tangencyPortfolio(returns): #returns has shape TxN
    if returns.shape[1] == 1:
        return 1
    else:
        mean_ret = np.mean(returns, axis=0,keepdims=True)
        cov_ret = np.cov(returns.T)
        weights = np.linalg.solve(cov_ret,mean_ret.T)
        return weights.T/np.sum(weights)


def preprocessMonthlyData(Data, _UNK = np.nan, normalizeCharacteristics = True, logdir=os.getcwd(), name = 'MonthlyData.npz'):
    if normalizeCharacteristics:
        DataChar = Data.iloc[:,[2]+list(range(5,Data.shape[1]))]
        grouped = DataChar.groupby('date')
        normalizedChar = grouped.transform(lambda x : x.rank(method='first')/x.count()-0.5)
        Data.iloc[:,5:] = normalizedChar
    else:
        name = name.replace(".npz", "Unnormalized.npz")

    savepath = os.path.join(logdir, name)
    if os.path.exists(savepath):
        logging.info("Monthly characteristics data already processed; skipping")
        return
    
    Data.index = pd.MultiIndex.from_frame(Data[['date','permno']])
    Data = Data.drop(columns = Data.columns[range(0,4)])
    Data.sort_index(inplace=True) 
    shape = Data.index.levshape + tuple([len(Data.columns)])
    data = np.full(shape, _UNK)
    data[tuple(Data.index.codes)] = Data.values
    
    date = Data.index.levels[0].to_numpy()
    permno = Data.index.levels[1].to_numpy()
    variable = Data.columns.to_numpy()

    np.save(os.path.join(logdir, "permnos.npy"), permno)
    np.savez(savepath, data = data, date = date, permno = permno, variable = variable)
    return 


def preprocessDailyReturns(Data, RiskFreeRates, adjustRF = True, _UNK = np.nan, logdir=os.getcwd(), name = 'DailyReturns-RFadjusted.npz'):
    savepath = os.path.join(logdir, name)
    if os.path.exists(savepath):
        logging.info("Daily returns already processed; skipping")
        return

    Data = Data[['date','permno','ret']]      
    Data.index = pd.MultiIndex.from_frame(Data[['date','permno']])
    Data = Data.drop(columns = Data.columns[range(0,2)])
    Data.sort_index(inplace=True) 
    shape = Data.index.levshape + tuple([len(Data.columns)])
    data = np.full(shape, _UNK)
    data[tuple(Data.index.codes)] = Data.values
    data = data[:,:,0]
    
    if adjustRF:
        data -= RiskFreeRates
    
    date = Data.index.levels[0].to_numpy()
    permno = Data.index.levels[1].to_numpy()

    # restrict returns data to only cover permnos that we have characteristics data for
    pmask = np.load(os.path.join(logdir, "permnos.npy"))
    data = data[:,np.isin(permno, pmask)]
    permno = pmask
    
    np.savez(savepath, data = data, date = date, permno = permno)
    return


class IPCA:
    def __init__(self, individual_feature_dim=46, logdir=os.getcwd(), debug=False,
                 pathMonthlyData = 'data/MonthlyData.npz', 
                 pathDailyData = 'data/DailyReturns-RFadjusted-new.npz', 
                 pathMonthlyDataUnnormalized = 'data/MonthlyDataUnnormalized.npz'):
        self._individual_feature_dim = individual_feature_dim #this is the number of characteristics, L
        self._logdir = logdir
        self._UNK = np.nan
        self._debug = debug
        
        monthlyData = np.load(pathMonthlyData, allow_pickle=True)
        dailyData = np.load(pathDailyData, allow_pickle=True)
        self.monthlyData = monthlyData['data']
        self.dailyData = dailyData['data']
        self.monthlyDataUnnormalized = np.load(pathMonthlyDataUnnormalized, allow_pickle=True)['data']
        self.monthlyCaps = np.nan_to_num(self.monthlyDataUnnormalized[:,:,19])
        
        self.dailyDates = pd.to_datetime(dailyData['date'], format='%Y%m%d')
        self.monthlyDates = pd.to_datetime(monthlyData['date'], format='%Y%m%d')
        # self.date2DailyIdx = {date:idx for idx, date in enumerate(self.dailyDates)}
        # self.date2MonthlyIdx = {date:idx for idx, date in enumerate(self.monthlyDates)}
        self.weight_matrices = []
        self.mask = np.zeros(0)
        self.weighted = False

    def _step_factor(self, R_list, I_list, Gamma, calculate_residual=False,startIndex=0):  # I are the characteristics Z in the paper
        f_list = []
        if calculate_residual:
            residual_list = []
        for (t, riTuple) in enumerate(zip(R_list, I_list)):
            R_t, I_t = riTuple
            beta_t = I_t.dot(Gamma)
            try:
                if self.weighted:
                    W_t = self.weight_matrices[t+startIndex]
                    A = beta_t.T @ W_t @ beta_t
                    # if not np.all(np.linalg.eigvals(A) > 0):
                    #    logging.info(f"beta.T @ W @ beta (shape {A.shape}) is not PSD!")
                    # if np.random.rand() > 0.99:
                    #     logging.info("Testing step_factor")
                    #     %timeit -n 5 sp.linalg.lstsq(A, beta_t.T @ W_t @ R_t)
                    #     %timeit -n 5 sp.linalg.solve(A, beta_t.T @ W_t @ R_t, assume_a="pos")
                    #     %timeit -n 5 sp.linalg.solve(A, beta_t.T @ W_t @ R_t, assume_a="sym")
                    #     %timeit -n 5 np.linalg.solve(A, beta_t.T @ W_t @ R_t)
                    #     %timeit -n 5 np.linalg.pinv(A) @ beta_t.T @ W_t @ R_t
                    f_t = np.linalg.solve(A, beta_t.T @ W_t @ R_t)
                else:
                    A = beta_t.T.dot(beta_t)
                    # if not np.all(np.linalg.eigvals(A) > 0):
                    #     logging.info(f"beta.T @ beta (shape {A.shape}) is not PSD!")
                    # if np.random.rand() > 0.99:
                    #     logging.info("Testing step_factor")
                    #     %timeit -n 5 sp.linalg.lstsq(beta_t, R_t)
                    #     %timeit -n 5 sp.linalg.solve(A, beta_t.T @ R_t, assume_a="pos")
                    #     %timeit -n 5 sp.linalg.solve(A, beta_t.T @ R_t, assume_a="sym")
                    #     %timeit -n 5 res1 = np.linalg.solve(A, beta_t.T @ R_t)
                    #     %timeit -n 5 np.linalg.pinv(A) @ beta_t.T @ R_t
                    #     %timeit -n 5 lr = LinearRegression(fit_intercept=False, n_jobs=-1); lr.fit(beta_t, R_t)
                    #     %timeit -n 5 chol,low = sp.linalg.cho_factor(A); res2 = sp.linalg.cho_solve((chol,low), beta_t.T @ R_t)
                    #     res1 = np.linalg.solve(A, beta_t.T @ R_t)
                    #     chol,low = sp.linalg.cho_factor(A); res2 = sp.linalg.cho_solve((chol,low), beta_t.T @ R_t)
                    #     logging.info("Chol vs np.linalg diff:", np.linalg.norm(res1 - res2, np.inf))
                    f_t = np.linalg.solve(A, beta_t.T.dot(R_t))
                    
            except np.linalg.LinAlgError as err:
                logging.info(str(err))
                if self.weighted:
                    W_t = self.weight_matrices[t+startIndex]
                    f_t = np.linalg.pinv(beta_t.T @ W_t @ beta_t).dot(beta_t.T @ W_t @ R_t)
                else:
                    f_t = np.linalg.pinv(beta_t.T.dot(beta_t)).dot(beta_t.T.dot(R_t))
            f_list.append(f_t)
            if calculate_residual:
                residual_list.append(R_t - beta_t.dot(f_t))
        if calculate_residual:
            return f_list, residual_list
        else:
            return f_list, None

    def _step_gamma(self, R_list, I_list, f_list, nFactors,startIndex=0):
        A = np.zeros((self._individual_feature_dim * nFactors, self._individual_feature_dim * nFactors))
        b = np.zeros((self._individual_feature_dim * nFactors, 1))
        for (t, rifTuple) in enumerate(zip(R_list, I_list, f_list)):
            R_t, I_t, f_t = rifTuple
            tmp_t = np.kron(I_t, f_t.T)
            if self.weighted:
                W_t = self.weight_matrices[t+startIndex]
                A += tmp_t.T @ W_t @ tmp_t
                b += tmp_t.T @ W_t @ R_t
            else:
                A += tmp_t.T.dot(tmp_t)
                b += tmp_t.T.dot(R_t)
        try:
            # if not np.all(np.linalg.eigvals(A) > 0):
            #     logging.info(f"sum(kron(z_t,f_t).T @ kron(z_t,f_t)) (shape {A.shape}) is not PSD!")
            # logging.info("Testing step_gamma")
            # %timeit -n 5 sp.linalg.lstsq(A, b)
            # %timeit -n 5 sp.linalg.solve(A, b, assume_a="pos")
            # %timeit -n 5 sp.linalg.solve(A, b, assume_a="sym")
            # %timeit -n 5 np.linalg.solve(A, b)
            # %timeit -n 5 np.linalg.pinv(A) @ b
            Gamma = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as err:
            logging.info(str(err))
            Gamma = np.linalg.pinv(A).dot(b)
        return Gamma.reshape((self._individual_feature_dim, nFactors))

    def _initial_gamma(self, R_list, I_list, nFactors, startIndex = 0):   #added by Jorge
        T = len(R_list)
        X = np.zeros((T, self._individual_feature_dim))
        for t in range(T):
            if self.weighted:
                W_t = self.weight_matrices[t+startIndex]
                X[t,:] = np.squeeze(I_list[t].T @ W_t @ R_list[t]) / len(R_list[t])
            else:
                X[t,:] = np.squeeze(I_list[t].T.dot(R_list[t])) / len(R_list[t])
        eValues, eVectors = np.linalg.eig(X.T.dot(X))
        Gamma = eVectors[:,:nFactors]
        return Gamma
 
    def _initial_factors(self, R_list, I_list, nFactors, startIndex = 0):
        T = len(R_list)
        X = np.zeros((T, self._individual_feature_dim))
        for t in range(T):
            if self.weighted:
                W_t = self.weight_matrices[t+startIndex]
                X[t,:] = np.squeeze(I_list[t].T @ W_t @ R_list[t]) / len(R_list[t])
            else:
                X[t,:] = np.squeeze(I_list[t].T.dot(R_list[t])) / len(R_list[t])
        pca = PCA(n_components=nFactors)
        pca.fit(X.T)  # pca.components_ matrix is of shape nFactors x T
        f_list = pca.components_
        return np.split(f_list, f_list.shape[1], axis=1)  
    
    def matrix_debug(self, matrix, name, extra=False):
        if not self._debug:
            return
        nanct = np.sum(np.isnan(matrix))
        zeroct = matrix.size - np.count_nonzero(matrix)
        nanpct = nanct/matrix.size * 100
        zeropct = zeroct/matrix.size * 100
        logging.info(f"'{name}' {matrix.shape}: has {matrix.size} entries and {nanct} NaNs ({nanpct:0.4f}%) and {zeroct} zeros ({zeropct:0.4f}%)")
        if extra:
            # also logging.info which rows/columns are mostly/all NaNs
            nanrowcts = np.sum(np.isnan(matrix), axis=1)  # nans per row
            nancolcts = np.sum(np.isnan(matrix), axis=0)  # nans per column
            nanrowct = np.sum(nanrowcts > 0)  # number of rows in which there is 1+ NaN entries
            nancolct = np.sum(nancolcts > 0)  # number of cols in which there is 1+ NaN entries
            nanrowpct = nanrowct/matrix.shape[0]*100
            nancolpct = nancolct/matrix.shape[1]*100
            logging.info(f"----> '{name}' {matrix.shape}: has {nanrowct} rows with NaNs ({nanrowpct:0.4f}%) and {nancolct} cols with NaNs ({nancolpct:0.4f}%)")
            nanidxs = np.argwhere(np.isnan(matrix))  # nan indices
            nancols = np.unique(nanidxs[:,1])  # column indices with nans
            # logging.info(f"'{name}' {matrix.shape}: NaN columns are {nancols}")
            # logging.info(f"'{name}' {matrix.shape}: NaN idxs are {nanidxs}")
            colnancts = {col: np.sum(nanidxs[:,1] == col) for col in nancols}
            logging.info(f"----> '{name}' {matrix.shape}: NaNs counts in column indices (idx : count) {colnancts}")
            
    def nanl2norm(self, x):
        return np.sqrt(np.nansum(np.square(x)))
    
    def compute_sparse_residuals(self, full_residuals_month, beta_month, R_month_clean, sparsity_quantile_threshold=0.01):
        """
        Computes a set of sparse residuals closest to those in full_residuals_month by 
        solving a sparse approximation problem for returns in R_month_clean using
        a candidate set of permnos given by information in beta_month.
        
        E.g. if 
            S = full_residuals_month in R(T x P)
            R = R_month_clean        in R(T x P)
            P = proj(beta_month)     in R(P x P)
        and
            S = R(1-P)
        then we want to estimate a sparse matrix
            C in R(P x P)
        such that C is close to P in some norm and
            S ~= RC
            
        Currently, we solve that problem by
            1. Hard thresholding I-P by only keeping values greater in magnitude than the 
               99th quantile
            2. Extracting a set of permnos from the thresholded I-P, the index mask 'h'
            3. Using this candidate set of entries in each row of P as selected variables
               to solve the regression for C[h,i], column i of matrix C restricted to rows in h:
                   S[:,i] = R[:,h] * C[h,i]
               I.e. all indices of C[:,i] are zero except for those in 'h', and we only
               select columns of R which are in 'h'.
            4. Running this regression for each col of S for which card(R[:,h]) > 1
            5. Letting the sparse residuals for the month be defined as C * R
            
        Future ideas include:
            - better sparse approximation/recovery than hard thresholding + linear regression 
                (simple lasso regression, variations on OMP, (F)ISTA, etc.)
            - denoising S_i prior to approximation (e.g. solving the above regression 
                minimizing the L1 norm of error instead of by L2 norm (e.g. by least squares))
            - using information in factor weighting matrix more carefully, e.g. by computing sparse eigenvectors
                via sparse PCA or sparse robust PCA and using those to get reconstructed, sparse
                (I-)P instead of thresholding (I-)P; (can sparse I-P be seen as negative graph lap?)
            - thresholding returns as well? how would we do this consistently through time? perhaps
                returns would need to be normalized (by rolling stats) first, but why would r_t+1 be 
                near zero if r_t was? this may not be a realistic direction to explore.
            - making portfolios more consistent through time:
                - add penalty s.t. 
                    |C(t-1) - C(t)|_1 <= f(|R(t-1)'R(t-1) - R(t)'R(t)|_2, |Z(t-1)'Z(t-1) - Z(t)'Z(t)|_2)
                    where f is some sum, product, or ratio of the changes in volatility amongst R and Z 
                    in adjacent time periods
                - initialize with an initial hard thresholding or sparse robust PCA reconstruction
                    or lasso regression of (I-)P
            - learn sparsification over time given trading policy
                - make proportional to drawdown?
                - select thresholding parameter/sparsity penalty based on SCEMPC branch n bound?
                - include interaction with testing for mean reversion
            - learn way to filter out outliers in distributions of characteristics, or use heuristic:
                - market cap filter
                - volume
                - learn which permnos will be around next month from function of characteristics in prior months
        """
        # (1) do hard thresholding
        eye = np.identity(beta_month.shape[0])
        proj = eye - beta_month @ np.linalg.pinv(beta_month.T @ beta_month) @ beta_month.T
        # thresholding should only be done wrt observed (nonzero) returns' indices' weights, 
        # so we set columns of proj to zero if more than half of the respective return column is missing
        # get idxs of columns where more than half of R_month_clean is zero
        mostly_zero_returns_col_idxs = np.argwhere(np.count_nonzero(R_month_clean, axis=0) <= R_month_clean.shape[0]//2).ravel()
        # set those columns in proj to zero
        proj[:,mostly_zero_returns_col_idxs] = 0
        threshold = np.quantile(np.abs(proj), 1 - sparsity_quantile_threshold)
        np.putmask(proj, np.abs(proj) < threshold, 0)  
        # mplot = plt.imshow((np.abs(proj) < threshold).T, interpolation='nearest', aspect='auto')
        # plt.savefig(f"test-sparse-proj-{beta_month.shape[1]}-{R_month_clean.shape[0]}-{R_month_clean.shape[1]}.png")
        # (2) print stats and such
        if np.random.rand() > 0.9:
            #print(proj.shape, R_month_clean.shape, beta_month.shape, full_residuals_month.shape)
            nonzero_portfolios = R_month_clean @ (eye - proj)
            num_nonzero_portfolios = np.sum(np.count_nonzero(nonzero_portfolios, axis=0) >= nonzero_portfolios.shape[0] * 0.9)
            #print(num_nonzero_portfolios, "portfolios (not containing self weight) left of", len(proj), 
                 # f"({num_nonzero_portfolios / len(proj) * 100:0.2f}% preserved)")
            nonzero_portfolios = R_month_clean @ proj
            num_nonzero_portfolios = np.sum(np.count_nonzero(nonzero_portfolios, axis=0) >= nonzero_portfolios.shape[0] * 0.9)
            #print(num_nonzero_portfolios, "portfolios left of", len(proj), 
                 # f"({num_nonzero_portfolios / len(proj) * 100:0.2f}% preserved)")
            ps = np.sum(np.abs(proj) > 0, axis=1)  # portfolio sizes
            ps = ps[ps > 0]
           # print("Size stats: ", "mean", np.mean(ps), "median", np.median(ps), "min", np.min(ps), 
                 # "max", np.max(ps), "q99", np.quantile(ps, 0.99), "num2to6:", np.sum((1 < ps) & (ps <= 6)))
            # the rows which include the permno itself (that is, the diagonal entry is nonzero)
            include_self_idxs = np.argwhere(np.diag(np.abs(proj)) > 1e-12).ravel()
            num_include_self = min(len(include_self_idxs), num_nonzero_portfolios)
            #print(f"Out of the {num_nonzero_portfolios} portfolios left, {num_include_self} include" + \
                 # f" themselves ({num_include_self/num_nonzero_portfolios*100:0.2f}%)")
            # out of the rows which include the permno itself, 
            # how many are comprised ONLY of the permno itself 
            # (e.g. they have one nonzero entry, which should be the permno's weight)
            num_just_self = np.sum(np.count_nonzero(proj[include_self_idxs,:], axis=1) == 1)
            #print(f"Out of the {num_nonzero_portfolios} portfolios left, {num_just_self} are " + \
                  #f"comprised of just one permno only: itself ({num_just_self/num_nonzero_portfolios*100:0.2f}%)")
        # (3) now perform regression and return sparse residuals
        C = np.zeros_like(proj)
        for i in range(proj.shape[0]):
            # get indices of nonzero weights
            h = np.argwhere(proj[:,i] != 0).ravel()
            if len(h) == 0:
                continue
            elif len(h) == 1:
                if h[0] == i:
                    C[i,i] = 1
                    continue
                else:
                    #logging.info("Hard thresholding got single weight which is not permno itself!")
                    continue
            Rh = R_month_clean[:,h]
            Si = full_residuals_month[:,i]
            if np.linalg.norm(Rh) <= 1e-8:
                pass
            C[h,i] = np.linalg.pinv(Rh.T.dot(Rh)).dot(Rh.T.dot(Si))
        return C
    
    def compute_weight_matrices(self, mask):
        """
        Computes a T-long sequence of NtxNt weight matrices Wt for the daily returns and mask provided,
        where Nt is the number of permnos in a given month and T is the number of months.
        Currently, only implements equal observation volatility weighting.
        """   
        weight_matrices = []
        for month in range(len(self.monthlyDates)):
            if month == 0:
                idxs_days_month = (self.dailyDates <= self.monthlyDates[0])
            else:
                idxs_days_month = (self.dailyDates > self.monthlyDates[month-1]) & (self.dailyDates <= self.monthlyDates[month])
            rmonth = self.dailyData[:,mask[month,:]][idxs_days_month,:]  # TtxNt matrix
            nans_per_col = np.count_nonzero(np.isnan(rmonth),axis=0)
            #nanszeros_per_col = np.count_nonzero(np.isnan(rmonth),axis=0) + np.sum(rmonth==0,axis=0)
            insufficient_data_mask = nans_per_col >= np.round(0.9 * rmonth.shape[0])
            vols = np.nanvar(rmonth, axis=0) #* rmonth.shape[0]  # Nt long vector
            vols[insufficient_data_mask] = np.nan 
            weight_mtx = np.diag(1/vols)
            weight_mtx[~np.isfinite(weight_mtx)] = 0
            weight_mtx[weight_mtx >= 10**6] = 0
            weight_matrices.append(weight_mtx)

        return weight_matrices
    
    def MonthlyOOSExpandingWindow(self, save=True, listFactors=list(range(1,21)), maxIter=1024, printOnConsole=True, printFreq=8, tol=1e-03,initialMonths=30*12,sizeWindow=24*12,CapProportion=0):
        R = self.monthlyData[:,:,0]
        I = self.monthlyData[:,:,1:]
        cap_chosen_idxs = self.monthlyCaps/np.nansum(self.monthlyCaps,axis=1,keepdims=True) >= CapProportion*0.01
        mask = (~np.isnan(R)) * cap_chosen_idxs
        R_reshape = np.expand_dims(R[mask], axis=1)
        I_reshape = I[mask]
        splits = np.sum(mask, axis=1).cumsum()[:-1]  #np.sum(mask, axis=1) how many stocks we have per year; the other cumukatively except for the last one
        R_list = np.split(R_reshape, splits)
        I_list = np.split(I_reshape, splits)
        nWindows = int((R.shape[0] - initialMonths)/sizeWindow)

        for nFactors in listFactors:
            residualsOOS = np.zeros_like(R[initialMonths:,:], dtype=float)
            for nWindow in range(nWindows):
                if nWindow == 0:
                    f_list = self._initial_factors(R_list[:initialMonths], I_list[:initialMonths], nFactors)
                    self._Gamma = np.zeros((self._individual_feature_dim, nFactors))
                    nIter = 0
                    while nIter < maxIter:
                        Gamma = self._step_gamma(R_list[:initialMonths], I_list[:initialMonths], f_list, nFactors)
                        f_list, _ = self._step_factor(R_list[:initialMonths], I_list[:initialMonths],Gamma)
                        nIter += 1
                        dGamma = np.max(np.abs(self._Gamma - Gamma))
                        self._Gamma = Gamma
                        if printOnConsole and nIter % printFreq == 0:
                            logging.info('nFactor: %d\t, nWindow: %d/%d\t, nIter: %d\t, dGamma: %0.2e' %(nFactors,nWindow,nWindows,nIter,dGamma))
                        if nIter > 1 and dGamma < tol:
                            break
                else:
                    nIter = 0
                    while nIter < maxIter:
                        f_list, _ = self._step_factor(R_list[:(initialMonths+nWindow*sizeWindow)], I_list[:(initialMonths+nWindow*sizeWindow)],self._Gamma)
                        Gamma = self._step_gamma(R_list[:(initialMonths+nWindow*sizeWindow)], I_list[:(initialMonths+nWindow*sizeWindow)], f_list, nFactors)
                        nIter += 1
                        dGamma = np.max(np.abs(self._Gamma - Gamma))
                        self._Gamma = Gamma
                        if printOnConsole and nIter % printFreq == 0:
                            logging.info('nFactor: %d\t, nWindow: %d/%d\t, nIter: %d\t, dGamma: %0.2e' %(nFactors,nWindow,nWindows,nIter,dGamma))
                        if nIter > 1 and dGamma < tol:
                            break
                    
                f_list, residual_list = self._step_factor(R_list[(initialMonths+nWindow*sizeWindow):(initialMonths+(nWindow+1)*sizeWindow)], I_list[(initialMonths+nWindow*sizeWindow):(initialMonths+(nWindow+1)*sizeWindow)], self._Gamma, calculate_residual=True)
                residualsOOS[(nWindow*sizeWindow):((nWindow+1)*sizeWindow),:][mask[(initialMonths+nWindow*sizeWindow):(initialMonths+(nWindow+1)*sizeWindow),:]] = np.squeeze(np.concatenate(residual_list))
                
            if printOnConsole:                
                logging.info('Finished! (nFactors = %d)' %nFactors)
            if save:
                np.save(os.path.join(self._logdir, 'IPCA_OOSresiduals_%d_factors_%d_initialMonths_%d_window.npy' %(nFactors,initialMonths,sizeWindow)), residualsOOS)
        return residualsOOS
        
    # CapProportion = 0.001, 0.01  #The betas are going to be constant each month, so essentially constant at the daily level (and the factors will change daily)
    # SizeWindow must divide  = months of Data - training months
        
    def DailyOOSRollingWindow(self, save=True, weighted=True, listFactors=list(range(1,21)), maxIter=1024, 
                                printOnConsole=True, printFreq=8, tol=1e-03, initialMonths=35*12, 
                                sizeWindow=15*12, CapProportion = 0.001, save_beta=False, save_gamma=False, 
                                save_rmonth=False, save_mask=False, save_sparse_weights_month=False,
                                skip_oos=False,reestimationFreq = 12):    
        matrix_debug = self.matrix_debug
        R = self.monthlyData[:,:,0]
        I = self.monthlyData[:,:,1:]
        cap_chosen_idxs = self.monthlyCaps/np.nansum(self.monthlyCaps,axis=1,keepdims=True) >= CapProportion*0.01
        mask = (~np.isnan(R)) * cap_chosen_idxs
        self.mask = mask
        if save_mask:
            mask_path = os.path.join(self._logdir, f"mask_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy")
            np.save(mask_path, mask)
        # with np.printoptions(threshold=np.inf):      
        #     print(np.count_nonzero(mask,axis=1))                  
        R_reshape = np.expand_dims(R[mask], axis=1)
        I_reshape = I[mask]
        splits = np.sum(mask, axis=1).cumsum()[:-1]  #np.sum(mask, axis=1) how many stocks we have per year; the other cumukatively except for the last one
        R_list = np.split(R_reshape, splits)
        I_list = np.split(I_reshape, splits)
        self.R_list = R_list
        self.I_list = I_list
        nWindows = int((R.shape[0] - initialMonths)/reestimationFreq)
        logging.info(f"nWindows {nWindows}")
        
        if weighted:
            self.weighted = True
            self.weight_matrices = self.compute_weight_matrices(mask)
        
        firstOOSDailyIdx = np.argmax(self.dailyDates >= (pd.datetime(self.dailyDates.year[0],self.dailyDates.month[0],1)+pd.DateOffset(months=initialMonths)))
        logging.info(f"firstidx {firstOOSDailyIdx}")
        logging.info(f"self.dailyData.shape[0] {self.dailyData.shape[0]}")
        Rdaily = self.dailyData[firstOOSDailyIdx:,:]
        sharpesFactors = np.zeros(len(listFactors))
        counter = 0
        
        DataTrain = np.load(os.path.join('residuals', 'ipca_normalized', f"IPCA_DailyOOSresiduals_1_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{reestimationFreq}_reestimationFreq_{CapProportion}_cap.npy"))
        assetsToConsider = np.count_nonzero(DataTrain, axis=0) >= 30 #chooses stocks which have at least #lookback non-missing observations in all the training time
        logging.info(np.where(assetsToConsider))
        logging.info(f"sum a2c {np.sum(assetsToConsider)}")
        Ntilde = np.sum(assetsToConsider)  # the residuals we are actually going to trade
        T,N = Rdaily.shape
        logging.info(f'N {N} Ntilde {Ntilde}')
        superMask = np.count_nonzero(mask[initialMonths-1:],axis=0)>=1
        Nsupertilde = np.sum(superMask)  # the maximum assets that are going to be involved in the interesting residuals
        logging.info(f"superMask {superMask.shape} {Nsupertilde} {len(superMask)}")
        np.save('residuals-new/super_mask.npy',superMask)
        
        if not os.path.isdir(self._logdir + "_stuff"):
            try:
                os.mkdir(self._logdir + "_stuff")
            except Exception as e:
                logging.info(f"Could not create folder '{self._logdir + '_stuff'}'!")
                raise e

        if printOnConsole:
            logging.info("Beginning daily residual computations")
        for nFactors in listFactors:
            residualsOOS = np.zeros_like(Rdaily, dtype=float)
            factorsOOS = np.zeros_like(Rdaily[:,:nFactors], dtype=float)
            sparse_oos_residuals = np.zeros_like(Rdaily, dtype=float)
            T,N=residualsOOS.shape
            residualsMatricesOOS = np.zeros((T,Ntilde,Nsupertilde), dtype=np.float32)
            
            # WeightsFactors = np.zeros((T,N,N))
            # WeightsSparseFactors = np.zeros((T,N,N))
            if nFactors == 0:
                for month in range((initialMonths),R.shape[0]):
                    idxs_days_month = (self.dailyDates[firstOOSDailyIdx:] >  self.monthlyDates[month-1]) \
                                    & (self.dailyDates[firstOOSDailyIdx:] <= self.monthlyDates[month])
                    R_month = Rdaily[:,mask[month-1,:]][idxs_days_month,:] #TxN
                    # change missing values to zeros to exclude them from calculation
                    R_month_clean = R_month.copy()
                    R_month_clean[np.isnan(R_month_clean)] = 0
                    residuals_month = R_month
                    # set residuals equal to zero wherever there are NaNs from missing returns, as (NaN - prediction) is still NaN
                    residuals_month[np.isnan(residuals_month)] = 0
                    sparse_residuals_month = residuals_month
                    temp = residualsOOS[:,mask[month-1,:]].copy()
                    temp[idxs_days_month,:] = residuals_month
                    residualsOOS[:,mask[month-1,:]] = temp
                    sparse_temp = sparse_oos_residuals[:,mask[month-1,:]].copy()
                    sparse_temp[idxs_days_month,:] = sparse_residuals_month
                    sparse_oos_residuals[:,mask[month-1,:]] = sparse_temp
            else:
                for nWindow in range(nWindows):         
                    # Load or estimate Gamma; use save_gamma=True to force estimation
                    # Gamma estimation
                    logging.info("Estimating gamma")
                    if nWindow == 0:
                        gamma_path = os.path.join(self._logdir+'_stuff', f"gamma_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{reestimationFreq}_reestimationFreq_{CapProportion}_cap_{nWindow}.npy")
                        if os.path.isfile(gamma_path) and not save_gamma:
                            Gamma = np.load(gamma_path)
                            self._Gamma = Gamma
                        else:
                            f_list = self._initial_factors(R_list[(initialMonths+nWindow*reestimationFreq-sizeWindow):(initialMonths+nWindow*reestimationFreq)], I_list[(initialMonths+nWindow*reestimationFreq-sizeWindow):(initialMonths+nWindow*reestimationFreq)], nFactors,startIndex=initialMonths+nWindow*reestimationFreq-sizeWindow)
                            self._Gamma = np.zeros((self._individual_feature_dim, nFactors))
                            nIter = 0
                            while nIter < maxIter:
                                Gamma = self._step_gamma(R_list[(initialMonths+nWindow*reestimationFreq-sizeWindow):(initialMonths+nWindow*reestimationFreq)], I_list[(initialMonths+nWindow*reestimationFreq-sizeWindow):(initialMonths+nWindow*reestimationFreq)], f_list, nFactors,startIndex=initialMonths+nWindow*reestimationFreq-sizeWindow)
                                f_list, _ = self._step_factor(R_list[(initialMonths+nWindow*reestimationFreq-sizeWindow):(initialMonths+nWindow*reestimationFreq)], I_list[(initialMonths+nWindow*reestimationFreq-sizeWindow):(initialMonths+nWindow*reestimationFreq)], Gamma,startIndex=initialMonths+nWindow*reestimationFreq-sizeWindow)
                                nIter += 1
                                dGamma = np.max(np.abs(self._Gamma - Gamma))
                                self._Gamma = Gamma
                                if printOnConsole and nIter % printFreq == 0:
                                    logging.info('nFactor: %d,\t nWindow: %d/%d,\t nIter: %d,\t dGamma: %0.2e' % (nFactors,nWindow,nWindows,nIter,dGamma))
                                if nIter > 1 and dGamma < tol:
                                    break
                            if save_gamma:                            
                                np.save(gamma_path, Gamma)
                    else:
                        gamma_path = os.path.join(self._logdir+'_stuff', f"gamma_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{reestimationFreq}_reestimationFreq_{CapProportion}_cap_{nWindow}.npy")
                        if os.path.isfile(gamma_path) and not save_gamma:
                            Gamma = np.load(gamma_path)
                            self._Gamma = Gamma
                        else:
                            nIter = 0
                            while nIter < maxIter/2:
                                f_list, _ = self._step_factor(R_list[(initialMonths+nWindow*reestimationFreq-sizeWindow):(initialMonths+nWindow*reestimationFreq)], I_list[(initialMonths+nWindow*reestimationFreq-sizeWindow):(initialMonths+nWindow*reestimationFreq)], self._Gamma,startIndex=initialMonths+nWindow*reestimationFreq-sizeWindow)
                                Gamma = self._step_gamma(R_list[(initialMonths+nWindow*reestimationFreq-sizeWindow):(initialMonths+nWindow*reestimationFreq)], I_list[(initialMonths+nWindow*reestimationFreq-sizeWindow):(initialMonths+nWindow*reestimationFreq)], f_list, nFactors,startIndex=initialMonths+nWindow*reestimationFreq-sizeWindow)
                                nIter += 1
                                dGamma = np.max(np.abs(self._Gamma - Gamma))
                                self._Gamma = Gamma
                                if printOnConsole and nIter % printFreq == 0:
                                    logging.info('nFactor: %d,\t nWindow: %d/%d,\t nIter: %d,\t dGamma: %0.2e' %(nFactors,nWindow,nWindows,nIter,dGamma))
                                if nIter > 1 and dGamma < tol:
                                    break
                            if save_gamma:
                                np.save(gamma_path, Gamma)

                    if not skip_oos:
                        # Computation of out-of-sample residuals   
                        for month in range((initialMonths+nWindow*reestimationFreq),(initialMonths+(nWindow+1)*reestimationFreq)):
                            if self._debug: logging.info(f"--- Month: {month}/{(initialMonths+(nWindow+1)*sizeWindow)} ----")
                            beta_month = I[month-1,mask[month-1,:]].dot(self._Gamma)  # N x nfactors
                            # self.matrix_debug(beta_month, "beta_month")
                            idxs_days_month = (self.dailyDates[firstOOSDailyIdx :] > self.monthlyDates[month-1]) \
                                            & (self.dailyDates[firstOOSDailyIdx:] <= self.monthlyDates[month])
                            R_month = Rdaily[:,mask[month-1,:]][idxs_days_month,:] #TxN
                            if save_rmonth:
                                r_path = os.path.join(self._logdir+'_stuff', f"rmonth_{month}_month_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{reestimationFreq}_reestimationFreq_{CapProportion}_cap.npy")
                                np.save(r_path, R_month)
                            # change missing values to zeros to exclude them from calculation
                            R_month_clean = R_month.copy()
                            R_month_clean[np.isnan(R_month_clean)] = 0
                            
                            # try:
                            #     if weighted:
                            #         W_month = self.weight_matrices[month-1]
                            #         factors_month = np.linalg.solve(beta_month.T @ W_month @ beta_month, beta_month.T @ W_month @ R_month_clean.T) # nfactors x T
                            #     else:
                            #         factors_month = np.linalg.solve(beta_month.T.dot(beta_month), beta_month.T.dot(R_month_clean.T)) # nfactors x T
                            #         #factors_month = np.linalg.solve(beta_month.T.dot(beta_month), beta_month.T.dot(R_month.T)) # nfactors x T
                            # except np.linalg.LinAlgError as err:
                            
                            if weighted:
                                factors_month = np.linalg.pinv(beta_month.T @ W_month @ beta_month).dot(beta_month.T @ W_month @ R_month_clean.T) # nfactors x T
                            else:
                                factors_month = np.linalg.pinv(beta_month.T @ beta_month).dot(beta_month.T @ R_month_clean.T) # nfactors x T
                            residuals_month = R_month - factors_month.T.dot(beta_month.T)
                            # set residuals equal to zero wherever there are NaNs from missing returns, as (NaN - prediction) is still NaN
                            # residuals_month2 = residuals_month.copy()
                            residuals_month[np.isnan(residuals_month)] = 0
                            # print(np.sum(residuals_month2*~np.isnan(R_month)),np.sum(np.isnan(residuals_month)))
                            # print(np.linalg.norm(residuals_month2*~np.isnan(R_month)-residuals_month))
                            
                            sparse_weights_month = self.compute_sparse_residuals(residuals_month, beta_month, R_month_clean)
                            sparse_residuals_month = R_month_clean @ sparse_weights_month
                            if save_sparse_weights_month:
                                sw_path = os.path.join(self._logdir+'_stuff', f"sparseweights_{month}_month_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{reestimationFreq}_reestimationFreq_{CapProportion}_cap.npy")
                                np.save(sw_path, sparse_weights_month)
                            if save_beta:
                                beta_path = os.path.join(self._logdir+'_stuff', f"beta_{month}_month_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{reestimationFreq}_reestimationFreq_{CapProportion}_cap.npy")
                                np.save(beta_path, beta_month)
                            #self.matrix_debug(residuals_month, "residuals_month", extra=True)
                            
                            temp = residualsOOS[:,mask[month-1,:]].copy()
                            temp[idxs_days_month,:] = residuals_month
                            residualsOOS[:,mask[month-1,:]] = temp
                            sparse_temp = sparse_oos_residuals[:,mask[month-1,:]].copy()
                            sparse_temp[idxs_days_month,:] = sparse_residuals_month
                            sparse_oos_residuals[:,mask[month-1,:]] = sparse_temp
                            factorsOOS[idxs_days_month,:] = factors_month.T
                                                         
                            Tprime,Nprime = R_month_clean.shape
                            MatrixFull = np.zeros((Tprime,N,N))
                            MatrixReduced = np.nan_to_num(np.eye(Nprime)-beta_month@np.linalg.pinv(beta_month.T @ beta_month)@beta_month.T).T # Nprime x Nprime
                            mask_month = np.broadcast_to(~np.isnan(R_month).reshape((Tprime,Nprime,1)),(Tprime,Nprime,Nprime))
                            # print(np.matmul(mask_month*np.broadcast_to(MatrixReduced,(Tprime,Nprime,Nprime)),R_month_clean.reshape((Tprime,Nprime,1))).squeeze().shape,residuals_month.shape)
                            # print(np.linalg.norm(np.matmul(mask_month*np.broadcast_to(MatrixReduced,(Tprime,Nprime,Nprime)),R_month_clean.reshape((Tprime,Nprime,1))).squeeze()-residuals_month))
                            # print(np.linalg.norm(np.nan_to_num(MatrixReduced@R_month.T-residuals_month.T)))
                            # print(MatrixFull[np.broadcast_to(mask[month-1,:].reshape((N,1))@mask[month-1,:].reshape((1,N)),(Tprime,N,N))].shape)
                            MatrixFull[np.broadcast_to(mask[month-1,:].reshape((N,1))@mask[month-1,:].reshape((1,N)),(Tprime,N,N))] = (mask_month*np.broadcast_to(MatrixReduced,(Tprime,Nprime,Nprime))).ravel()
                            # print(np.linalg.norm(np.matmul(MatrixFull,np.nan_to_num(Rdaily[idxs_days_month].reshape((Tprime,N,1)))).squeeze()-residualsOOS[idxs_days_month]))
                            residualsMatricesOOS[idxs_days_month] = MatrixFull[:,assetsToConsider][:,:,superMask]
                            # print(residualsMatricesOOS[idxs_days_month].shape)
                            portfolio1 = residualsOOS[idxs_days_month][:,assetsToConsider]
                            portfolio2 = np.matmul(residualsMatricesOOS[idxs_days_month], np.nan_to_num(Rdaily[idxs_days_month][:,superMask]).reshape(Tprime,Nsupertilde,1)).squeeze()
                            transition_error = np.linalg.norm(portfolio1 - portfolio2) / Tprime
                            logging.info(transition_error)
                            # logging.info('New month!')

                            # MatrixFull = np.zeros((N,N))
                            # MatrixFull[mask[month-1,:].reshape((N,1))@mask[month-1,:].reshape((1,N))] = np.nan_to_num(sparse_weights_month.T.ravel())
                            # residualsMatricesOOS[idxs_days_month] = np.broadcast_to(MatrixFull[assetsToConsider][:,assetsToConsider],(np.sum(idxs_days_month),Ntilde,Ntilde))
                            # print(np.linalg.norm(sparse_oos_residuals[idxs_days_month][:,assetsToConsider]-np.matmul(residualsMatricesOOS[idxs_days_month],np.nan_to_num(Rdaily[idxs_days_month][:,assetsToConsider],copy=False).reshape(Tprime,Ntilde,1)).squeeze())/Tprime)    
                            
                            # if printOnConsole and random.random() > 0.9:
                            #     rel_approx_error = self.nanl2norm(temp - sparse_temp) / self.nanl2norm(temp)
                            #     print(f"----> Sparse portfolio relative approximation error is {rel_approx_error*100:0.4f}%")
                            #     self.matrix_debug(residualsOOS, "residualsOOS")
                            #     self.matrix_debug(sparse_oos_residuals, "sparse_oos_residuals")
                
                if not skip_oos:
                    factorsOOS = np.nan_to_num(factorsOOS)
                    sharpesFactors[counter] = get_sharpe_tangencyPortfolio(factorsOOS)
                    counter += 1
                    
            if printOnConsole:                
                logging.info('Finished (nFactors = %d)' %nFactors)
            if save and not skip_oos:
                rsavepath = os.path.join(self._logdir, f"IPCA_DailyOOSresiduals_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy")
                msavepath = os.path.join(self._logdir, f"IPCA_DailyMatrixOOSresiduals_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy")
                logging.info(f"Saving {rsavepath}")
                np.save(rsavepath, residualsOOS)
                # np.save(os.path.join(self._logdir, f"IPCA_sparseDailyOOSresiduals_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy"), sparse_oos_residuals)
                logging.info(f"Saving {msavepath}")
                np.save(msavepath, residualsMatricesOOS)
                # np.save(os.path.join(self._logdir, f"IPCA_sparseDailyMatrixOOSresiduals_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy"), sparse_oos_residuals)                
        if not skip_oos:
            pass
            # np.savetxt(os.path.join(self._logdir,f"IPCA_DailyOOSFactorsSharpes_{initialMonths}_initialMonths_{sizeWindow}_window_{reestimationFreq}_reestimationFreq_{CapProportion}_cap.csv"),sharpesFactors,delimiter=',',header = f"Factors: {listFactors}")
            
        return
    
    def DailyOOSExpandingWindow(self, save=True, weighted=True, listFactors=list(range(1,21)), maxIter=1024, 
                                printOnConsole=True, printFreq=8, tol=1e-03, initialMonths=30*12, 
                                sizeWindow=24*12, CapProportion = 0.001, save_beta=False, save_gamma=False, 
                                save_rmonth=False, save_mask=False, save_sparse_weights_month=False,
                                skip_oos=False):    
        matrix_debug = self.matrix_debug
        R = self.monthlyData[:,:,0]
        I = self.monthlyData[:,:,1:]
        cap_chosen_idxs = self.monthlyCaps/np.nansum(self.monthlyCaps,axis=1,keepdims=True) >= CapProportion*0.01
        mask = (~np.isnan(R)) * cap_chosen_idxs
        self.mask = mask
        if save_mask:
            mask_path = os.path.join(self._logdir, f"mask_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy")
            np.save(mask_path, mask)
        with np.printoptions(threshold=np.inf):
            logging.info(np.count_nonzero(mask,axis=1))                  
        R_reshape = np.expand_dims(R[mask], axis=1)
        I_reshape = I[mask]
        splits = np.sum(mask, axis=1).cumsum()[:-1]  #np.sum(mask, axis=1) how many stocks we have per year; the other cumukatively except for the last one
        R_list = np.split(R_reshape, splits)
        I_list = np.split(I_reshape, splits)
        self.R_list = R_list
        self.I_list = I_list
        nWindows = int((R.shape[0] - initialMonths)/sizeWindow)
        logging.info(f"nWindows {nWindows}")
        
        if weighted:
            self.weighted = True
            self.weight_matrices = self.compute_weight_matrices(mask)
        
        firstOOSDailyIdx = np.argmax(self.dailyDates >= (pd.datetime(self.dailyDates.year[0],self.dailyDates.month[0],1)+pd.DateOffset(months=initialMonths)))
        logging.info(f"firstidx {firstOOSDailyIdx}")
        logging.info(f"self.dailyData.shape[0] {self.dailyData.shape[0]}")
        Rdaily = self.dailyData[firstOOSDailyIdx:,:]
        sharpesFactors = np.zeros(len(listFactors))
        counter = 0
        
        if not os.path.isdir(self._logdir + "_stuff"):
            try:
                os.mkdir(self._logdir + "_stuff")
            except Exception as e:
                logging.info(f"Could not create folder '{self._logdir + '_stuff'}'!")
                raise e

        if printOnConsole:
            logging.info("Beginning daily residual computations")
        for nFactors in listFactors:
            residualsOOS = np.zeros_like(Rdaily, dtype=float)
            factorsOOS = np.zeros_like(Rdaily[:,:nFactors], dtype=float)
            sparse_oos_residuals = np.zeros_like(Rdaily, dtype=float)
            T,N=residualsOOS.shape
            #WeightsFactors = np.zeros((T,N,N))
            #WeightsSparseFactors = np.zeros((T,N,N))
            for nWindow in range(nWindows):
                if nFactors == 0:
                    for month in range((initialMonths+nWindow*sizeWindow),(initialMonths+(nWindow+1)*sizeWindow)):
                        idxs_days_month = (self.dailyDates[firstOOSDailyIdx :] > self.monthlyDates[month-1]) \
                                            & (self.dailyDates[firstOOSDailyIdx:] <= self.monthlyDates[month])
                        R_month = Rdaily[:,mask[month-1,:]][idxs_days_month,:] #TxN
                        # change missing values to zeros to exclude them from calculation
                        R_month_clean = R_month.copy()
                        R_month_clean[np.isnan(R_month_clean)] = 0
                        residuals_month = R_month
                        # set residuals equal to zero wherever there are NaNs from missing returns, as (NaN - prediction) is still NaN
                        residuals_month[np.isnan(residuals_month)] = 0
                        sparse_residuals_month = residuals_month
                        temp = residualsOOS[:,mask[month-1,:]].copy()
                        temp[idxs_days_month,:] = residuals_month
                        residualsOOS[:,mask[month-1,:]] = temp
                        sparse_temp = sparse_oos_residuals[:,mask[month-1,:]].copy()
                        sparse_temp[idxs_days_month,:] = sparse_residuals_month
                        sparse_oos_residuals[:,mask[month-1,:]] = sparse_temp
                    
                else:          
                    # Load or estimate Gamma; use save_gamma=True to force estimation
                    gamma_path = os.path.join(self._logdir+'_stuff', f"gamma_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy")
                    if os.path.isfile(gamma_path) and not save_gamma:
                        Gamma = np.load(gamma_path)
                        self._Gamma = Gamma
                    # Gamma estimation
                    else:
                        logging.info("Estimating gamma")
                        if nWindow == 0:
                            f_list = self._initial_factors(R_list[:initialMonths], I_list[:initialMonths], nFactors)
                            self._Gamma = np.zeros((self._individual_feature_dim, nFactors))
                            nIter = 0
                            while nIter < maxIter:
                                Gamma = self._step_gamma(R_list[:initialMonths], I_list[:initialMonths], f_list, nFactors)
                                f_list, _ = self._step_factor(R_list[:initialMonths], I_list[:initialMonths], Gamma)
                                nIter += 1
                                dGamma = np.max(np.abs(self._Gamma - Gamma))
                                self._Gamma = Gamma
                                if printOnConsole and nIter % printFreq == 0:
                                    logging.info('nFactor: %d,\t nWindow: %d/%d,\t nIter: %d,\t dGamma: %0.2e' %(nFactors,nWindow,nWindows,nIter,dGamma))
                                if nIter > 1 and dGamma < tol:
                                    break
                        else:
                            nIter = 0
                            while nIter < maxIter:
                                f_list, _ = self._step_factor(R_list[:(initialMonths+nWindow*sizeWindow)], I_list[:(initialMonths+nWindow*sizeWindow)], self._Gamma)
                                Gamma = self._step_gamma(R_list[:(initialMonths+nWindow*sizeWindow)], I_list[:(initialMonths+nWindow*sizeWindow)], f_list, nFactors)
                                nIter += 1
                                dGamma = np.max(np.abs(self._Gamma - Gamma))
                                self._Gamma = Gamma
                                if printOnConsole and nIter % printFreq == 0:
                                    logging.info('nFactor: %d,\t nWindow: %d/%d,\t nIter: %d,\t dGamma: %0.2e' %(nFactors,nWindow,nWindows,nIter,dGamma))
                                if nIter > 1 and dGamma < tol:
                                    break
                        if save_gamma:
                            gamma_path = os.path.join(self._logdir+'_stuff', f"gamma_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy")
                            np.save(gamma_path, Gamma)

                    if not skip_oos:
                        # Computation of out-of-sample residuals   
                        for month in range((initialMonths+nWindow*sizeWindow),(initialMonths+(nWindow+1)*sizeWindow)):
                            if self._debug: logging.info(f"--- Month: {month}/{(initialMonths+(nWindow+1)*sizeWindow)} ----")
                            beta_month = I[month-1,mask[month-1,:]].dot(self._Gamma)  # N x nfactors
                            #self.matrix_debug(beta_month, "beta_month")
                            idxs_days_month = (self.dailyDates[firstOOSDailyIdx :] > self.monthlyDates[month-1]) \
                                            & (self.dailyDates[firstOOSDailyIdx:] <= self.monthlyDates[month])
                            R_month = Rdaily[:,mask[month-1,:]][idxs_days_month,:] #TxN
                            if save_rmonth:
                                r_path = os.path.join(self._logdir+'_stuff', f"rmonth_{month}_month_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy")
                                np.save(r_path, R_month)
                            # change missing values to zeros to exclude them from calculation
                            R_month_clean = R_month.copy()
                            R_month_clean[np.isnan(R_month_clean)] = 0
                            try:
                                if weighted:
                                    W_month = self.weight_matrices[month-1]
                                    factors_month = np.linalg.solve(beta_month.T @ W_month @ beta_month, beta_month.T @ W_month @ R_month_clean.T) # nfactors x T
                                else:
                                    factors_month = np.linalg.solve(beta_month.T.dot(beta_month), beta_month.T.dot(R_month_clean.T)) # nfactors x T
                                    #factors_month = np.linalg.solve(beta_month.T.dot(beta_month), beta_month.T.dot(R_month.T)) # nfactors x T
                            except np.linalg.LinAlgError as err:
                                logging.info(f"----> Linear algebra error: {str(err)}")
                                if weighted:
                                    factors_month = np.linalg.pinv(beta_month.T @ W_month @ beta_month).dot(beta_month.T @ W_month @ R_month.T) # nfactors x T
                                else:
                                    factors_month = np.linalg.pinv(beta_month.T @ beta_month).dot(beta_month.T @ R_month.T) # nfactors x T
                            residuals_month = R_month - factors_month.T.dot(beta_month.T)
                            # set residuals equal to zero wherever there are NaNs from missing returns, as (NaN - prediction) is still NaN
                            residuals_month[np.isnan(residuals_month)] = 0
                            sparse_weights_month = self.compute_sparse_residuals(residuals_month, beta_month, R_month_clean)
                            sparse_residuals_month = R_month_clean @ sparse_weights_month
                            if save_sparse_weights_month:
                                sw_path = os.path.join(self._logdir+'_stuff', f"sparseweights_{month}_month_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy")
                                np.save(sw_path, sparse_weights_month)
                            if save_beta:
                                beta_path = os.path.join(self._logdir+'_stuff', f"beta_{month}_month_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy")
                                np.save(beta_path, beta_month)
                            #self.matrix_debug(residuals_month, "residuals_month", extra=True)
                            temp = residualsOOS[:,mask[month-1,:]].copy()
                            temp[idxs_days_month,:] = residuals_month
                            residualsOOS[:,mask[month-1,:]] = temp
                            sparse_temp = sparse_oos_residuals[:,mask[month-1,:]].copy()
                            sparse_temp[idxs_days_month,:] = sparse_residuals_month
                            sparse_oos_residuals[:,mask[month-1,:]] = sparse_temp
                            factorsOOS[idxs_days_month,:] = factors_month.T

                            if printOnConsole:
                                #rel_approx_error = self.nanl2norm(temp - sparse_temp) / self.nanl2norm(temp)
                                #logging.info(f"----> Sparse portfolio relative approximation error is {rel_approx_error*100:0.4f}%")
                                self.matrix_debug(residualsOOS, "residualsOOS")
                                self.matrix_debug(sparse_oos_residuals, "sparse_oos_residuals")
                
                if not skip_oos:
                    factorsOOS = np.nan_to_num(factorsOOS)
                    sharpesFactors[counter] = get_sharpe_tangencyPortfolio(factorsOOS)
                    counter += 1
                    
            if printOnConsole:                
                logging.info('Finished! (nFactors = %d)' %nFactors)
            if save and not skip_oos:
                rsavepath = os.path.join(self._logdir, f"IPCA_DailyOOSresiduals_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy")
                msavepath = os.path.join(self._logdir, f"IPCA_DailyMatrixOOSresiduals_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy")
                logging.info(f"Saving {rsavepath}")
                np.save(rsavepath, residualsOOS)
                # np.save(os.path.join(self._logdir, f"IPCA_sparseDailyOOSresiduals_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy"), sparse_oos_residuals)
                # logging.info(f"Saving {msavepath}")
                # np.save(msavepath, residualsMatricesOOS)
                #np.save(os.path.join(self._logdir, f"IPCA_sparseDailyMatrixOOSresiduals_{nFactors}_factors_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.npy"), sparse_oos_residuals)                
        
        if not skip_oos:
            pass
            # np.savetxt(os.path.join(self._logdir,f"IPCA_DailyOOSFactorsSharpes_{initialMonths}_initialMonths_{sizeWindow}_window_{CapProportion}_cap.csv"),sharpesFactors,delimiter=',',header = f"Factors: {listFactors}")
            
        return