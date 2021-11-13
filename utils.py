import os
import datetime
import socket
import logging
import multiprocessing
from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np
import yaml
import petname
from gpuinfo import GPUInfo
from twilio.rest import Client


def initialize_logging(app_name, logdir="logs", debug=False, run_id=None):
    debugtag = "-debug" if debug else ""
    logtag = petname.Generate(2)
    username = os.path.split(os.path.expanduser("~"))[-1]
    hostname = socket.gethostname().replace(".stanford.edu","")
    if not os.path.isdir(logdir):
        os.mkdir(logdir)
    starttimestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if run_id is not None:
        raise Exception("Unimplemented")
        run_id = str(run_id)
        fh = logging.FileHandler(f"{logdir}/{app_name}{debugtag}_{run_id}_{logtag}_{username}_{hostname}_{starttimestr}.log")
        ch = logging.StreamHandler()
        # create formatter and add it to the handlers
        formatter = logging.Formatter(f"[%(asctime)s] Run{run_id} - %(levelname)s - %(message)s", '%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logging.getLogger('').handlers = []
        logging.getLogger('').addHandler(fh)
        logging.getLogger('').addHandler(ch)
        logging.info(f"HANDLER LEN: {len(logging.getLogger('').handlers)}")
        logging.getLogger('').level = logging.INFO if not debug else logging.DEBUG
    else:
        logging.basicConfig(
            level=logging.INFO if not debug else logging.DEBUG,
            format="[%(asctime)s] %(levelname)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(f"{logdir}/{app_name}{debugtag}_{logtag}_{username}_{hostname}_{starttimestr}.log"),
                logging.StreamHandler()
            ]
        )
    logging.info(f"Logging initialized for '{app_name}' by '{username}' on host '{hostname}' with ID '{logtag}'")
    return username, hostname, logtag, starttimestr
        
    
def slides_barplot(labels: list, data_by_group: dict):
    """
    Example input:
        labels = ['FF', 'PCA', 'IPCA', 'SDF']
        data_by_group['1'] = [20, 34, 30, 35]
        data_by_group['3'] = [25, 32, 34, np.nan]
    """
    # ax, fig = plt.subplot()

    # get width of bar
    barWidth = 0.25
     
    # set height of bar
    bars1 = [12, 30, 1, 8, 22]
    bars2 = [28, 6, 16, 5, 10]
    bars3 = [29, 3, 24, 25, 17]
     
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
     
    # Make the plot
    plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='var1')
    plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='var2')
    plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')
     
    # Add xticks on the middle of the group bars
    plt.xlabel('group', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], ['A', 'B', 'C', 'D', 'E'])
     
    # Create legend & Show graphic
    plt.legend()
    plt.show()


def nploadp(filepath, 
            blocksize=1024,  # tune this for performance/granularity
            log=True
           ):
    """
    Load Numpy array with progress messages. Log progress using logging if log=True.
    """
    try:
        mmap = np.load(filepath, mmap_mode='r')
        y = np.empty_like(mmap)
        n_blocks = int(np.ceil(mmap.shape[0] / blocksize))
        for b in range(n_blocks):
            if log:
                logging.info('Loading Numpy array into memory, block {}/{}'.format(b+1, n_blocks))
            else:
                nowdatestr = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{nowdatestr}] Loading Numpy array into memory, block {b+1}/{n_blocks}")
            y[b*blocksize : (b+1) * blocksize] = mmap[b*blocksize : (b+1) * blocksize]
    finally:
        del mmap  # make sure file is closed again
    return y


def get_free_gpu_ids(min_memory_mb=4000, gpu_memory_capacity_mb=12000):
    """
    Returns a list of integer IDs of GPUs with at least `min_memory_mb` memory free
    """
    # get GPU utilization (first row volatile GPU utilization % in [0,100], second row memory used in [0,12000])
    mem_free = gpu_memory_capacity_mb - np.array(GPUInfo.gpu_usage())  # 2 x num_GPUs
    # copy GPU indices into first row
    mem_free[0,:] = np.arange(mem_free.shape[1])
    # sort column of (GPU index, free memory) by free memory
    mem_free = mem_free[:,mem_free[1,:].argsort(kind='stable')[::-1]]
    # return indices which have more than min_memory_mb, sorted in order of most free memory
    return mem_free[0,np.where(mem_free[1,:] >= min_memory_mb)[0]].tolist()


def send_twilio_message(message: str, to: str):
    """
    `to` number must be in the form of "+1234567890" for e.g. +1 123 456 7890
    Find your Account SID and Auth Token at twilio.com/console, see http://twil.io/secure.
    Put credentials into `credentials.yaml` file in main directory.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        with open(os.path.join(dir_path, "credentials.yaml"), 'r') as f:
            credentials = yaml.load(f)
    except Exception as e:
        logging.warning("Could not load credentials.yaml file", exc_info=e)
        return
    try:
        account_sid = credentials['TWILIO_ACCOUNT_SID']
        auth_token = credentials['TWILIO_AUTH_TOKEN']
        from_phone_number = credentials['TWILIO_PHONE_NUMBER']
        client = Client(account_sid, auth_token)
        message = client.messages.create(body=message, from_=from_phone_number, to=to)
    except Exception as e:
        logging.warning("Could not load send Twilio message", exc_info=e)
        return

def moving_average(a, n=30, axis=1):
    """
    Creates a simple moving average for the matrix `a`, with a window size of `n`, along the given `axis`.
    """
    ret = np.cumsum(a,axis)
    N = a.shape[axis]
    if axis == 0:
        ret[n:] = ret[n:] - ret[:-n] # np.take(ret, range(n,N), axis=axis) - np.take(ret, range(0,N-n), axis=axis, out = )
        try:
            ret[:n - 1] = ret[:n-1] / np.linspace(1,min(n-1,N),min(n-1, N))
        except:
            ret[:n - 1] = ret[:n-1] / np.linspace(1,min(n-1,N),min(n-1, N)).reshape((min(n-1,N),1))
        ret[n - 1:] = ret[n - 1:] / n
    elif axis == 1:
        ret[:,n:] = ret[:,n:] - ret[:,:-n] # np.take(ret, range(n,N), axis=axis) - np.take(ret, range(0,N-n), axis=axis, out = )
        ret[:,:n - 1] = ret[:,:n-1] / np.linspace(1,min(n-1,N),min(n-1, N)).reshape((min(n-1,N),1))
        ret[:,n - 1:] = ret[:,n - 1:] / n
    else:
        raise Exception(f"Invalid axis for moving average '{axis}'")
    return ret


def import_string(dotted_path):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    From https://docs.djangoproject.com/en/dev/_modules/django/utils/module_loading/.
    """
    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
    except ValueError as err:
        raise ImportError("%s doesn't look like a module path" % dotted_path) from err

    module = import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError as err:
        raise ImportError('Module "%s" does not define a "%s" attribute/class' % (
            module_path, class_name)
        ) from err

