Deep Learning Statistical Arbitrage
===================================

This repo contains the official code for our paper *Deep Learning Statistical Arbitrage*, available at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3862004 and https://arxiv.org/abs/2106.04028.

## Quickstart

To test a trading policy model on a residual time series, use `run_train_test.py`. 
This file exports a function, `run()`, which can be imported and used in e.g. a
grid search, or run from the command line. Command line usage will suit most users.
To run from the command line, use
```
python3 run_train_test.py -c configs/config_name_here.yaml
```
where `config_name_here.yaml` is a configuration file from the `configs` folder.
You can write your own configuration file to edit hyperparameters and other 
settings for the trading test. See `run_train_test.py` for other command line options.

## Structure

This repo is organized as follows:
- `train_test.py` contains the code for training a trading policy model and simulating trading.
- `run_train_test.py` is a user interface to `train_test.py` which deals with configuration, logging, saving results, etc.
- `preprocess.py` contains functions for preprocessing residual time series data into a form usable by a trading policy model
- `data.py` contains miscellaneous functions for altering residual time series data
- `config` contains configuration files which define various tests of trading policy models on residual time series
- `data` should contain raw input data used to create residuals
- `factor_models` contains code for creating residuals from raw input data
- `residuals` stores residual time series data sets created by the code in `factor_models`
- `models` contains code for trading policy models
- `results` will contain the results of and plots for trading policy model tests conducted by `run_train_test.py`
- `logs` will contain logs for runs of models and factor models
- `tools` should contain miscellaneous code for interpreting and exploring results and saved models
- `utils.py` contains helpful functions used throughout

## Generating residuals

To create residuals, first ensure that input data is present in the `data` directory, then run `run_factor_model.py`, providing the name of a factor model in the `factor_models` directory, e.g.
```
python3 run_factor_model.py -m factor_model_name_here
```
Generated residuals for the factor model will be saved in the `residuals` folder.

## Contributing

Code is released as is, but we welcome pull requests for any issues. 

## Notes

Note that `use_residual_weights` must be set to True in configuration files to reproduce results of the paper. Unfortunately, we can't release original asset return and characteristic data due to licensing agreements with our data providers. The corresponding author for this work is Greg Zanotti.
