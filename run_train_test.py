import argparse
import datetime
import json
import logging
import gzip
import os
import pathlib
import pickle
import re
import shutil
import socket
import time

import yaml
import petname
import numpy as np
import pandas as pd
import torch

from train_test import test, estimate
from data import perturb
from preprocess import *
from models import *
from utils import initialize_logging, nploadp, import_string, get_free_gpu_ids, send_twilio_message

torch.set_default_dtype(torch.float)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = False
    

def configure_logging(app_name:str, run_id:str = None, logdir:str = "logs", debug=False):
    debugtag = "-debug" if debug else ""
    run_id = str(run_id)
    username = os.path.split(os.path.expanduser("~"))[-1]
    hostname = socket.gethostname().replace(".stanford.edu","")
    if not os.path.isdir(logdir):
        os.mkdir(logdir)
    starttimestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logtag = petname.Generate(2)
    
    fh = logging.FileHandler(f"{logdir}/{app_name}{debugtag}_{run_id}_{logtag}_{username}_{hostname}_{starttimestr}.log")
    ch = logging.StreamHandler()
    # create formatter and add it to the handlers
    formatter = logging.Formatter(f"[%(asctime)s] Run-{run_id} - %(levelname)s - %(message)s", '%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logging.getLogger('').handlers = []
    logging.getLogger('').addHandler(fh)
    logging.getLogger('').addHandler(ch)
    logging.info(f"STARTED LOGGING FROM CHILD")
    return username, hostname, logtag, starttimestr


def run(config:dict, 
        run_id:str = None,
        gpu_device_ids:list = None,
        notification_phone_number:str = None,):
    """
    Runs a test of a trading policy model over a residual time series using the configuration `config`.

    If run_id is given, logging messages and results files will incorporate it (useful when automating calls to run() for e.g. grid search)
    If gpu_device_ids is None, GPUs will automatically be selected. Set to a list of ints to use those device IDs (useful when automating calls to run() for e.g. grid search).
    If notification_phone_number is given and Twilio is set up, phone number will be sent an SMS upon completion of or exception in a trading policy test.
    """
    model_name = config['model_name']
    results_tag = config['results_tag']
    debug = config['debug']
    username, hostname, log_tag, starttime = configure_logging(model_name, run_id=run_id, debug=debug) \
                                            if run_id else initialize_logging(model_name, debug=debug) 
    
    try:
        # TODO: add current git hash to config
        # use https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
        logging.info(f"Config: \n{json.dumps(config, indent=2, sort_keys=False)}")
        
        results_filename = f"results_{log_tag}_{results_tag}"
        factor_models = config['factor_models']
        cap = config['cap_proportion']
        use_residual_weights = config['use_residual_weights']
        objective = config['objective']

        # set up data
        filepaths = []
        residual_weightsNames = []
        datanames = []
        results_dict = {}
        #IPCA
        ipcadir = "ipca_normalized"
        ipcartag = "IPCA_DailyOOSresiduals"
        ipcamtag = "IPCA_DailyMatrixOOSresiduals"
        for factor in factor_models["IPCA"]:
            im = 420 #initial months
            w = 20*12 #window size
            filepaths += [f"residuals/{ipcadir}/{ipcartag}_{factor}_factors_{im}_initialMonths_{w}_window_12_reestimationFreq_{cap}_cap.npy"]
            datanames += ['IPCA'+str(factor)]
            residual_weightsNames += [f"residuals/{ipcadir}/{ipcamtag}_{factor}_factors_{im}_initialMonths_{w}_window_12_reestimationFreq_{cap}_cap.npy"]
        #PCA
        pcadir = "pca"
        pcartag = "AvPCA_OOSresiduals"
        pcamtag = "AvPCA_OOSMatrixresiduals"
        for factor in factor_models["PCA"]:
            ioy = 1998
            w = 60
            cw = 252
            filepaths += [f"residuals/{pcadir}/{pcartag}_{factor}_factors_{ioy}_initialOOSYear_{w}_rollingWindow_{cw}_covWindow_{cap}_Cap.npy"]
            datanames += ['PCA'+str(factor)]
            residual_weightsNames += [f"residuals/{pcadir}/{pcamtag}_{factor}_factors_{ioy}_initialOOSYear_{w}_rollingWindow_{cw}_covWindow_{cap}_Cap.npy"]
        #FamaFrench
        ffdir = "famafrench"
        ffrtag = "DailyFamaFrench_OOSresiduals"
        ffmtag = "DailyFamaFrench_OOSMatrixresiduals"
        for factor in factor_models["FamaFrench"]:
            ioy = 1998
            w = 60 
            filepaths += [f"residuals/{ffdir}/{ffrtag}_{factor}_factors_{ioy}_initialOOSYear_{w}_rollingWindow_{cap}_Cap.npy" ]
            #filepaths += [f"residuals/ff-universe-residuals/ff-universe-residuals_{factor}_factors_1998_initialOOSYear_60_rollingWindow_0.01_Cap.npy" ]
            datanames += ['FamaFrench'+str(factor)]
            #datanames += ['FamaFrenchNew'+str(factor)]
            residual_weightsNames += [f"residuals/{ffdir}/{ffmtag}_{factor}_factors_{ioy}_initialOOSYear_{w}_rollingWindow_{cap}_Cap.npy" ]
            #residual_weightsNames += [f"residuals/ff-universe-residuals/ff-universe-transition-matrices_{factor}_factors_1998_initialOOSYear_60_rollingWindow_0.01_Cap.npy" ]

        # load dates
        dates_filepath = 'data/F-F_Research_Data_5_Factors_2x3_daily.CSV'
        if not os.path.exists(dates_filepath):
            ff5 = pd.read_csv("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", header=2, index_col=0)
            ff5.to_csv(dates_filepath)
        FamaFrenchDailyData = pd.read_csv(dates_filepath, index_col=0) / 100
        daily_dates = pd.to_datetime(
            FamaFrenchDailyData.index[(FamaFrenchDailyData.index > 19980000) & (FamaFrenchDailyData.index < 20170000)],
            format ='%Y%m%d')
        del FamaFrenchDailyData
    
        # Test loop
        for i in range(len(filepaths)):
            # TODO: modify config dict for each factor model and #factors so that results are saved with unique config
            logging.info(f'Testing {filepaths[i]}')
            filepath = filepaths[i]
            logging.info('Loading residuals')
            if not os.path.exists(filepath) and os.path.exists(filepath + ".gz"):
                logging.info("Unzipping residual file")
                with gzip.open(filepath + ".gz", 'rb') as f_in:
                    with open(filepath, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            residuals = np.load(filepath).astype(np.float32)
            if 'perturbation' in config and len(config['perturbation']) > 0:
                logging.info(f"Before perturbing residuals: std: {np.std(residuals[residuals != 0]):0.4f}")
                residuals = perturb(residuals, config['perturbation'])
                logging.info(f"After perturbing residuals: std: {np.std(residuals[residuals != 0]):0.4f}")
            logging.info('Residuals loaded')
            residuals[np.isnan(residuals)] = 0

            if use_residual_weights:
                logging.info('Loading residual composition matrix')
                residual_weight_marker = "__residual_weights"
                # residual_weights = np.load(residual_weightsNames[i]) #.astype(np.float32)
                residual_weights = nploadp(residual_weightsNames[i])
                logging.info('Residual composition matrix loaded')
            else:
                residual_weight_marker = ""
                residual_weights = None

            if objective not in ["sharpe", "meanvar", "sqrtMeanSharpe"]:
                raise Exception(f"Invalid objective '{objective}'")

            # define model and preprocess function
            model = import_string(f"models.{config['model_name']}.{config['model_name']}")
            preprocess = import_string(f"preprocess.{config['preprocess_func']}")
            model_tag = datanames[i] \
                            + f"__{config['model_name']}" \
                            + residual_weight_marker \
                            + f"__{objective}" \
                            + f"__{config['trans_cost']}trans_cost" \
                            + f"__{config['hold_cost']}hold_cost" \
                            + f"__{config['model']['lookback']}lookback" \
                            + f"__{config['length_training']}length_training" \
                            + (f"__{results_tag}" if results_tag != "" else "") \
                            + f"__{log_tag}" \
                            + ""
                            # + "".join([f"__{config['model'][k]}{k}" for k in config['model'] if k != 'lookback']) \
                            # + f"__{int(time.time())}" \
            logging.info('STARTING: ' + model_tag)
            
            if gpu_device_ids is None:
                if   config['model']['lookback'] == 30 and config['length_training'] == 1000:
                    num_gpus_needed = 3
                elif config['model']['lookback'] == 30 and config['length_training'] >= 2000:
                    num_gpus_needed = 3
                elif config['model']['lookback'] == 60 and config['length_training'] == 1000:
                    num_gpus_needed = 4
                else:
                    logging.error("Unknown context for estimating number of GPUs needed for training")
                    num_gpus_needed = int(input("Enter number of GPUs needed for model (integer):"))
                    logging.info(f"User entered '{num_gpus_needed}' GPUs needed for this model's training")
                device_ids = get_free_gpu_ids(min_memory_mb=9000)[:num_gpus_needed]
            else:
                device_ids = gpu_device_ids

            # prepare output folder            
            outdir = os.path.join(str(pathlib.Path().resolve()), 'results', config['model_name'])
            if not os.path.exists(outdir):
                os.makedirs(outdir)
                
            # TODO: pre-allocate CUDA memory, or loop trying until we can.
            # incrementally increase num_gpus_needed when we can't allocate all the needed memory
            # following is useful:
            # https://discuss.pytorch.org/t/reserving-gpu-memory/25297
            # https://gist.github.com/sparkydogX/845b658e3e6cef58a7bf706a9f43d7bf
            # log for debugging:  torch.cuda.memory_summary(device=None, abbreviated=False)
            logging.info(f"Running on devices {device_ids}")
            if config['mode'] == 'test':
                rets,sharpe,ret,std,turnover,short_proportion = test(residuals, 
                                                                     daily_dates,
                                                                     model,
                                                                     preprocess,
                                                                     config,
                                                                     residual_weights = residual_weights,
                                                                     save_params = True,
                                                                     force_retrain = config['force_retrain'],
                                                                     parallelize = True, 
                                                                     log_dev_progress_freq = 10, 
                                                                     log_plot_freq = 149, 
                                                                     device = f'cuda:{device_ids[0]}', 
                                                                     device_ids = device_ids,
                                                                     output_path = outdir, 
                                                                     num_epochs = config['num_epochs'], 
                                                                     early_stopping = config['early_stopping'],
                                                                     model_tag = model_tag, 
                                                                     batchsize = config['batch_size'], 
                                                                     retrain_freq = config['retrain_freq'], 
                                                                     rolling_retrain = config['rolling_retrain'],
                                                                     length_training = config['length_training'], 
                                                                     lookback = config['model']['lookback'], 
                                                                     trans_cost = config['trans_cost'], 
                                                                     hold_cost = config['hold_cost'],
                                                                     objective = config['objective'],
                                                                     )
            elif config['mode'] == 'estimate':  # used to train a model once, e.g. for hyperparameter exploration on dev dataset
                rets,sharpe,ret,std,turnover,short_proportion = estimate(residuals, 
                                                                         daily_dates,
                                                                         model,
                                                                         preprocess,
                                                                         config,
                                                                         residual_weights = residual_weights,
                                                                         save_params = True,
                                                                         force_retrain = config['force_retrain'],
                                                                         parallelize = True, 
                                                                         log_dev_progress_freq = 10, 
                                                                         log_plot_freq = 149, 
                                                                         device = f'cuda:{device_ids[0]}', 
                                                                         device_ids = device_ids,
                                                                         output_path = outdir, 
                                                                         num_epochs = config['num_epochs'], 
                                                                         lr = config['learning_rate'], 
                                                                         early_stopping = config['early_stopping'],
                                                                         model_tag = model_tag, 
                                                                         batchsize = config['batch_size'], 
                                                                         length_training = config['length_training'], 
                                                                         test_size = config['retrain_freq'], 
                                                                         lookback = config['model']['lookback'], 
                                                                         trans_cost = config['trans_cost'], 
                                                                         hold_cost = config['hold_cost'],
                                                                         objective = config['objective'],
                                                                         )
            else:
                raise Exception(f"Invalid mode '{config['mode']}'; must be either 'test' or 'estimate'")

            results_dict[model_tag] = {
                "returns": rets,
                "sharpe": sharpe,
                "ret": ret,
                "std": std,
                "turnover": turnover,
                "short_proportion": short_proportion,
                "config": config,
                "timestamp": datetime.datetime.now()
            }
            # TODO: move results saving to MongoDB
            pkl_filename = f'results/{model_name}/{results_filename}'
            if os.path.exists(pkl_filename + ".pickle"):
                pkl_filename += str(int(time.time())) + ".pickle"
            else:
                pkl_filename += ".pickle"
            with open(pkl_filename, 'wb') as handle:
                pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logging.error("Uncaught exception", exc_info=e)
        if notification_phone_number:
            error_msg = f"FAILED: {results_tag} - {model_name} - {log_tag} - {hostname} - {starttime} - {repr(e)}"
            send_twilio_message(error_msg, notification_phone_number)
        raise e
    if notification_phone_number:
        completion_msg = f"COMPLETED: {results_tag} - {model_name} - {log_tag} - {hostname} - {starttime}"
        send_twilio_message(completion_msg, notification_phone_number)
    

def init_argparse():
    parser = argparse.ArgumentParser(
        description="Test trading policy model on residual time series given configuration file."
    )
    parser.add_argument("--config", "-c", help="path to a .yaml configuration file (e.g. 'config/cnntransformer-full.yaml')", required=True)
    parser.add_argument("--run-id", "-r", help="identifier string carrying external information (e.g. 'run42')", required=False)
    parser.add_argument("--gpu-device-ids", "-g", nargs="*", type=int, help="space-separated list of GPU device IDs to use (e.g. '0 1 2 3')", required=False)
    parser.add_argument("--notification-phone-number", "-p", help="notification phone number string (e.g. '+12345678900')", required=False)
    return parser


def main():
    # TODO: add support for multiple configs and parallelization of runs across GPUs (using pytorch.multiprocessing)
    parser = init_argparse()
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            args.config = config
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)
    print("Running...")
    run(**vars(args))


if __name__ == "__main__":
    main()
