import argparse
import logging
import os

import numpy as np
import pandas as pd

from factor_models.famafrench import FamaFrench
from factor_models.pca import PCA
from factor_models.ipca import IPCA, preprocessDailyReturns
from utils import initialize_logging


def run_ipca():
    initialize_logging("IPCA")
    logging.info("Loading characteristics data")
    MonthlyData = pd.read_csv("data/CharAll_na_rm.csv")
    logging.info("Loading daily returns")
    DailyData = pd.read_csv('data/daily-returns-clean.csv')
    logging.info("Loading risk-free rates")
    RiskFreeData = pd.read_csv('data/F-F_Research_Data_Factors_daily.CSV')
    RFs = RiskFreeData.loc[(RiskFreeData['Date'] > 19630000) & (RiskFreeData['Date'] < 20170000),['RF']].to_numpy() / 100
    logging.info("Preprocessing monthly characteristics data")
    preprocessDailyReturns(MonthlyData, logdir='data')
    logging.info("Preprocessing daily returns")
    preprocessDailyReturns(DailyData, RFs, logdir='data')
    logging.info("Initializing IPCA factor model")
    ipca = IPCA(logdir=os.path.join('residuals-new', 'ipca_normalized'))
    for capProportion in [0.01]:#, 0.001]:
        for sizeWindow in [20*12]:# 15*12]:
            logging.info(f"Running IPCA for window size {sizeWindow}, cap proportion {capProportion}")
            ipca.DailyOOSRollingWindow(listFactors=[0,1,3,5,8,10,15],
                                        initialMonths=420,
                                        sizeWindow=sizeWindow,
                                        CapProportion=capProportion,
                                        maxIter=1500,
                                        weighted=False,
                                        save=True,
                                        save_beta=False,
                                        save_gamma=False, 
                                        save_rmonth=False, 
                                        save_mask=False,
                                        save_sparse_weights_month=False,
                                        skip_oos=False,
                                        reestimationFreq = 12)


def run_pca():
    pca = PCA(logdir=os.path.join('residuals', 'pca'))
    for cap in [0.01]: #,0.001]:
        pca.OOSRollingWindowPermnos(CapProportion = cap, save=True,sizeWindow=60,sizeCovarianceWindow=252,factorList=[0,1,3,5,8,10,15])
        #pca.OOSRollingWindowPermnosVectorized(CapProportion = cap, save=True,sizeWindow=60,sizeCovarianceWindow=252,factorList=range(0,16))


def run_famafrench():
    ff5 = pd.read_csv("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", header=2, index_col=0)
    mom = pd.read_csv("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip", header=11, index_col=0)
    strev = pd.read_csv("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_ST_Reversal_Factor_daily_CSV.zip", header=11, index_col=0)
    ltrev = pd.read_csv("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_LT_Reversal_Factor_daily_CSV.zip", header=11, index_col=0)
    factors = [ff5,mom,strev,ltrev]

    for i in range(len(factors)):
        factors[i].index = factors[i].index.map(str)

    ff8 = pd.concat(factors, axis=1, join="inner")
    ff8.index = ff8.index.astype('int')
    ff8.rename(columns={col:col.strip() for col in ff8.columns}, inplace=True)
    ff8.drop(columns=['RF'], inplace=True)
    ff8.to_csv("data/FamaFrench8Daily.csv")

    FamaFrench = FamaFrench(logdir=os.path.join('residuals', 'famafrench'))
    for cap in [0.01]: 
        FamaFrench.OOSRollingWindowPermnos(cap=cap, save=True, sizeWindow=60, listFactors=[0,1,3,5,8])


def init_argparse():
    parser = argparse.ArgumentParser(
        description="Run factor model to create residuals from raw returns."
    )
    parser.add_argument("--model", "-m", help="the name of a factor model ('ipca', 'pca', or 'famafrench')", required=True)
    return parser


def main():
    parser = init_argparse()
    args = parser.parse_args()
    print("Running...")
    if args.model == "ipca":
        run_ipca()
    elif args.model == "pca":
        run_pca()
    elif args.model == "famafrench":
        run_famafrench()
    else:
        raise Exception(f"Invalid factor model '{args.model}'")


if __name__ == "__main__":
    main()
