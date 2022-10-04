#!/usr/bin/env python3

#Install the below libaries before importing
import pandas as pd
import dtale 
from pandas_profiling import ProfileReport


#EDA using pandas-profiling
# profile = ProfileReport(
#     pd.read_csv('/com.docker.devenvironments.code/data/Macro.csv').sample(frac=0.25),
#      explorative=True
#      )

# #Saving results to a HTML file
# profile.to_file("Macro_EDA_pandas_profiling.html")

#EDA using pandas-profiling
profile = ProfileReport(
    pd.read_csv('/com.docker.devenvironments.code/data/RetChar.csv').sample(frac=0.025),
    explorative=True
    )

#Saving results to a HTML file
profile.to_file("RetChar_EDA_pandas_profiling.html")