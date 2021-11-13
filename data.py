import numpy as np
import torch

torch.set_default_dtype(torch.float)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = False
    

def perturb(data, pconfig):
    """
    Takes in TxN NumPy residuals matrix and a perturbation configuration.
    Perturbs residuals matrix to add noise.
    """
    if pconfig['per_residual']:
        # compute std of each residual
        # trick: replace zeros with NaNs using np.where and use np.nanstd to except zeros from std calculation
        # this is done because zeros code for missing data in our dataset
        data_std = np.nanstd(np.where(np.isclose(data,0), np.nan, data), axis=0)
        # some residuals are all zero, and np.nanstd makes the resultant std nan, so set those std's back to zero, as they should be
        data_std[np.isnan(data_std)] = 0
    else:
        data_std = np.std(data[data != 0])
    if pconfig['noise_type'] == "gaussian":
        noise = pconfig['noise_mean'] + pconfig['noise_std_pct'] * data_std * np.random.randn(*(data.shape))
    else:
        raise Exception(f"Unimplemented noise type '{pconfig['noise_type']}'")
    # don't add noise to missing observations
    if pconfig['noise_only']:
        data = (data != 0) * noise
    else:
        data += (data != 0) * noise
    return data