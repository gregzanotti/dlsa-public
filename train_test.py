import logging
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from utils import import_string

torch.set_default_dtype(torch.float)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = True
# torch.use_deterministic_algorithms()
    
    
def train(model,
          preprocess,
          data_train, 
          data_dev = None, 
          log_dev_progress = True, log_dev_progress_freq = 50, log_plot_freq = 50,
          num_epochs = 100, lr = 0.001, batchsize = 200, optimizer_name = "Adam", optimizer_opts = {"lr": 0.001},
          early_stopping = False, early_stopping_max_trials=5, lr_decay = 0.5, 
          residual_weights_train = None, residual_weights_dev = None,
          save_params = True, output_path = None, model_tag = '', 
          lookback = 30, 
          trans_cost = 0, hold_cost = 0,
          parallelize = True, device = None, device_ids=[0,1,2,3,4,5,6,7],  # must use device='cuda' to parallelize
          force_retrain = True,
          objective = "sharpe",): 
        
    if output_path is None: output_path = model.logdir
    if device is None: device = model.device
    logging.info(f"train(): data_train.shape {data_train.shape}")
    
    # preprocess data
    # assets_to_trade chooses assets which have at least `lookback` non-missing observations in the training period
    # this does not induce lookahead bias because idxs_selected is backward-looking and 
    # will only select assets with at least `lookback` non-missing obs
    assets_to_trade = np.count_nonzero(data_train, axis=0) >= lookback
    logging.info(f"train(): assets_to_trade.shape {assets_to_trade.shape}")
    data_train = data_train[:,assets_to_trade]
    if residual_weights_train is not None: 
        residual_weights_train = residual_weights_train[:,assets_to_trade]
    T,N = data_train.shape
    logging.info(f"train(): T {T} N {N}")
    windows, idxs_selected = preprocess(data_train, lookback)
    logging.info(f"train(): windows.shape {windows.shape} idxs_selected.shape {idxs_selected.shape}")
    
    # start to train
    if parallelize:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device) 
    model.train()
    optimizer_func = import_string(f"torch.optim.{optimizer_name}")
    optimizer = optimizer_func(model.parameters(), **optimizer_opts)
                                     
    min_dev_loss = np.inf
    patience = 0
    trial = 0
    
    already_trained = False
    checkpoint_fname = f'Checkpoint-{model.module.random_seed if parallelize else model.random_seed}_seed_'+model_tag+'.tar'
    if os.path.isfile(os.path.join(output_path, checkpoint_fname)) and not force_retrain:
        already_trained = True
        checkpoint = torch.load(os.path.join(output_path, checkpoint_fname), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.train()
        logging.info('Already trained!')
    
    #train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    
    begin_time = time.time()
    for epoch in range(num_epochs):
        rets_full = np.zeros(T-lookback)
        short_proportion = np.zeros(T-lookback)
        turnover = np.zeros(T-lookback)
        
        # break input data up into batches of size `batchsize` and train over them
        for i in range(int((T-lookback)/batchsize)+1):
            weights= torch.zeros( (min(batchsize*(i+1),T-lookback)-batchsize*i, N), device=device)
            if epoch == 0 and i == 0:
                logging.info(f"epoch {epoch} batch {i} weights.shape {weights.shape}")
            else:
                logging.debug(f"epoch {epoch} batch {i} weights.shape {weights.shape}")
            logging.debug("stats: " +\
                f"idxs_selected.shape {idxs_selected.shape}, " +\
                f"filtered for batch {i} idxs_selected.shape {idxs_selected[batchsize*i:min(batchsize*(i+1),T-lookback),:].shape}, " +\
                f"weights.shape {weights.shape}, " +\
                f"batch period len {min(batchsize*(i+1),T) - batchsize*i}"
            )
            # weights[idxs_selected[batchsize*i:min(batchsize*(i+1),T-lookback),:]] = model(torch.tensor(windows[batchsize*i:min(batchsize*(i+1),T-lookback)][idxs_selected[batchsize*i:min(batchsize*(i+1),T-lookback),:]],device=device)) 
            idxs_batch_i = idxs_selected[batchsize*i:min(batchsize*(i+1),T-lookback),:]  # idxs of valid residuals to trade in batch i
            input_data_batch_i = windows[batchsize*i:min(batchsize*(i+1),T-lookback)][idxs_batch_i]  
            logging.debug(f"epoch {epoch} batch {i} input_data_batch_i.shape {input_data_batch_i.shape}")
            weights[idxs_batch_i] = model(torch.tensor(input_data_batch_i, device=device))
            if residual_weights_train is None:
                abs_sum = torch.sum(torch.abs(weights),axis=1,keepdim=True)
            else:  # residual_weights_train is TxN1xN2 (multiplied by returns on the right gives residuals)
                assert(weights.shape == residual_weights_train[lookback+batchsize*i:min(lookback+batchsize*(i+1),T),:,0].shape)
                T1,N1 = weights.shape  # weights is T1xN1
                weights2 = torch.bmm(weights.reshape(T1,1,N1), \
                                     torch.tensor(residual_weights_train[lookback+batchsize*i:min(lookback+batchsize*(i+1),T)],
                    device=device)).squeeze()  # will be T1xN2: weights2 is in underlying asset space
                if epoch == 0 and i == 0:
                    logging.info(f"epoch {epoch} batch {i} weights2.shape {weights2.shape}")
                else:
                    logging.debug(f"epoch {epoch} batch {i} weights2.shape {weights2.shape}")
                abs_sum = torch.sum(torch.abs(weights2),axis=1,keepdim=True)
                try: weights2 = weights2/abs_sum
                except: weights2 = weights2/(abs_sum + 1e-8)
            try: weights = weights/abs_sum
            except: weights = weights/(abs_sum + 1e-8)
            
            rets_train = torch.sum(weights*torch.tensor(data_train[lookback+batchsize*i:min(lookback+batchsize*(i+1),T),:],device=device),axis=1) 
            
            # # no minibatch
            # if quantile is None:
            #     weights= torch.zeros((T-lookback,N),device=device)
            #     #print(windows[idxs_selected].shape, weights[idxs_selected].shape)
            #     #breakpoint()        
            #     weights[idxs_selected] = model(torch.tensor(windows[idxs_selected],device=device)) 
            #     abs_sum = torch.sum(torch.abs(weights),axis=1,keepdim=True)
            #     weights= weights/(abs_sum+0.000000001)
            #     #weights= torch.zeros((T-lookback,N),device=device)
            #     #weights[abs_sum>0] = weights[abs_sum>0]/abs_sum[abs_sum>0].unsqueeze(1)
            # else: #test this 
            #     weights = torch.full((T-lookback,N),float('nan'),device=device)
            #     weights[idxs_selected] = model(torch.tensor(windows[idxs_selected],device=device))
            #     quantilesTop = torch.nanquantile(weights,1-quantile,axis=1)
            #     quantilesBottom = torch.nanquantile(weights,quantile,axis=1)
            #     weights= torch.zeros((T-lookback,N),device=device)
            #     weights[(weights>quantilesTop) * (weights<quantilesBottom)] = weights[(weights>quantilesTop) * (weights<quantilesBottom)]
            #     weights= weights/(torch.sum(torch.abs(weights),axis=1)+0.0000001)
            # rets_train = torch.sum(weights*torch.tensor(data_train[lookback:,:],device=device),axis=1) 

            # # sequential computation of weights
            # for t in range(lookback,data_dev.shape[0]):
            #     #idxs_selected = ~np.any(data_train[(t-lookback):t,:] == 0, axis = 0).ravel() 
            #     #inputs = np.cumsum(data_train[(t-lookback):t,idxs_selected],axis=0).T #(N,T)
            #     #inputs = torch.tensor(inputs,device=device)
            #     #weights = model(inputs)
            #     weights = model(torch.tensor(windows[t-lookback,idxs_selected[t-lookback,:],:],device=device))
            #     abs_sum = torch.sum(torch.abs(weights))
            #     if abs_sum > 0:
            #         weights = weights/abs_sum         
            #     rets_train = torch.sum(weights*torch.tensor(data_train[t,idxs_selected[t-lookback,:]],device=device)).reshape([1])
            #     if t == lookback:
            #         rets_train = rets_train
            #     else:
            #         rets_train = torch.cat((rets_train, rets_train))
            # #print(weights,weights.shape)        
            # #print(abs_sum.shape)
            
            if residual_weights_train is None:
                rets_train = rets_train \
                  - trans_cost * torch.cat(
                        (torch.zeros(1, device=device),
                         torch.sum(torch.abs(weights[1:] - weights[:-1]), axis=1))) \
                  - hold_cost * torch.sum(torch.abs(torch.min(weights, torch.zeros(1, device=device))), axis=1)
            else:
                rets_train = rets_train \
                    - trans_cost * torch.cat(
                        (torch.zeros(1,device=device),
                         torch.sum(torch.abs(weights2[1:] - weights2[:-1]), axis=1))) \
                    - hold_cost * torch.sum(torch.abs(torch.min(weights2, torch.zeros(1, device=device))), axis=1)
                
            mean_ret = torch.mean(rets_train)
            std = torch.std(rets_train)
            if objective == "sharpe":
                loss = -mean_ret/std
            elif objective == "meanvar":
                loss = -mean_ret*252 + std*15.9
            elif objective == "sqrtMeanSharpe":
                loss = -torch.sign(mean_ret)*np.sqrt(np.abs(mean_ret))/std
            else:
                raise Exception(f"Invalid objective loss {objective}")
            
            if not already_trained and ((parallelize and model.module.is_trainable) or (not parallelize and model.is_trainable)):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
            
            if residual_weights_train is None: 
                weights = weights.detach().cpu().numpy()
            else: 
                weights = weights2.detach().cpu().numpy()    
          
            rets_full[batchsize*i:min(batchsize*(i+1),T-lookback)] = rets_train.detach().cpu().numpy()
            turnover[batchsize*i:(min(batchsize*(i+1),T-lookback)-1)] = np.sum(np.abs(weights[1:]-weights[:-1]),axis=1)
            turnover[min(batchsize*(i+1),T-lookback)-1] = turnover[min(batchsize*(i+1),T-lookback)-2]  # just to simplify things
            short_proportion[batchsize*i:min(batchsize*(i+1),T-lookback)] = np.sum(np.abs(np.minimum(weights,0)),axis=1)
           
        if log_dev_progress and epoch % log_dev_progress_freq == 0:
            dev_loss_description = ""
            if data_dev is not None:
                rets_dev,dev_loss,dev_sharpe,dev_turnovers,dev_short_proportions,weights_dev,a2t = \
                    get_returns(model,
                                preprocess = preprocess,
                                objective = objective,
                                data_test = data_dev,
                                device = device, 
                                lookback = lookback, 
                                trans_cost = trans_cost, hold_cost = hold_cost, 
                                residual_weights = residual_weights_dev,)
                model.train()
                dev_mean_ret = np.mean(rets_dev)
                dev_std = np.std(rets_dev)
                dev_turnover  = np.mean(dev_turnovers)
                dev_short_proportion = np.mean(dev_short_proportions)
                dev_loss_description =  f", dev loss {-dev_loss:0.2f}, " \
                                        f"dev Sharpe: {-dev_sharpe*np.sqrt(252):0.2f}, " \
                                        f"ret: {dev_mean_ret*252:0.4f}, " \
                                        f"std: {dev_std*np.sqrt(252) :0.4f}, " \
                                        f"turnover: {dev_turnover:0.3f}, " \
                                        f"short proportion: {dev_short_proportion:0.3f}\n"
               
            full_ret = np.mean(rets_full)
            full_std = np.std(rets_full)
            full_sharpe = full_ret/full_std
            full_turnover = np.mean(turnover)
            full_short_proportion = np.mean(short_proportion)
            
            logging.info(f'Epoch: {epoch}/{num_epochs}, ' \
                         f'train Sharpe: {full_sharpe*np.sqrt(252):0.2f}, ' \
                         f'ret: {full_ret*252:0.4f}, ' \
                         f'std: {full_std*np.sqrt(252):0.4f}, ' \
                         f'turnover: {full_turnover:0.3f}, ' \
                         f'short proportion: {full_short_proportion:0.3f} \n' \
                          '       ' \
                         f' time per epoch: {(time.time()-begin_time)/(epoch+1):0.2f}s' \
                         + dev_loss_description)   
            
            if early_stopping:
                if dev_loss < min_dev_loss:
                    patience = 0
                    min_dev_loss = dev_loss
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                    }
                    torch.save(checkpoint, os.path.join(output_path, f'Checkpoint-{model.random_seed}_seed_{model_tag}.tar'))
                else:
                    patience += 1
                    if trial == early_stopping_max_trials:
                        logging.info('Early stopping max trials reached')
                        break
                    else: # reduce learning rate
                        trial += 1
                        logging.info('Reducing learning rate')
                        lr = optimizer.param_groups[0]['lr'] * lr_decay
                        checkpoint = torch.load(os.path.join(output_path,\
                                                f'Checkpoint-{model.random_seed}_seed_{model_tag}.tar'),
                                                map_location=device)    
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model = model.to(device)
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        model.train()
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        patience = 0
        
#         if epoch == num_epochs-1 and data_dev is not None and log_dev_progress: # or (epoch % log_plot_freq == 0)
#             #cum_rets_train = np.cumprod(1+rets_train.detach().cpu().numpy())
#             plt.figure()
#             cum_rets_train = np.cumprod(1+rets_full)
#             cum_rets_dev = np.cumprod(1+rets_dev)
#             plt.plot(cum_rets_train,label='Train')
#             plt.plot(cum_rets_dev, label='Dev')
#             plt.title('Cumulative returns')
#             plt.legend()
#             plt.show() 
            
#             plt.figure()
#             plt.plot(turnover,label='Train')
#             plt.plot(dev_turnovers, label='Dev')
#             plt.title('Turnover')
#             plt.legend()
#             plt.show() 
            
#             plt.figure()
#             plt.plot(short_proportion,label='Train')
#             plt.plot(dev_short_proportions, label='Dev')
#             plt.title('Short proportion')
#             plt.legend()
#             plt.show() 
            
        if already_trained: break
                                  
    if save_params and not already_trained:
        # can also save model.state_dict() directly w/o the dictionary; extension should then be .pth instead of .tar
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        } 
        checkpoint_fname = f'Checkpoint-{model.module.random_seed if parallelize else model.random_seed}_seed_'+model_tag+'.tar'
        torch.save(checkpoint, os.path.join(output_path, checkpoint_fname))
                                                                        
    logging.info(f'Training done - Model: {model_tag}, seed: {model.module.random_seed if parallelize else model.random_seed}')             
    if data_dev is not None:
        return rets_dev, dev_turnovers, dev_short_proportions, weights_dev, a2t
    else:
        return rets_full, turnover, short_proportion, weights, assets_to_trade
                

def get_returns(model, 
                preprocess,
                objective,
                data_test, 
                lookback=30, 
                trans_cost = 0, 
                hold_cost = 0, 
                residual_weights = None,
                load_params = False, 
                paths_checkpoints = [None], 
                device = None, 
                parallelize=False, 
                device_ids=[0,1,2,3,4,5,6,7],):
    
    if device is None: device = model.device
    if parallelize: model = nn.DataParallel(model, device_ids=device_ids).to(device)
        
    # restrict to assets which have at least `lookback` non-missing observations in the training period
    assets_to_trade = np.count_nonzero(data_test,axis=0) >= lookback
    logging.debug(f"get_returns(): assets_to_trade.shape {assets_to_trade.shape}")
    data_test = data_test[:,assets_to_trade]
    T,N = data_test.shape
    windows, idxs_selected = preprocess(data_test, lookback)   
    
    rets_test = torch.zeros(T-lookback)
    #weightsTest = torch.zeros(N,device='cpu')
    #weightsComplete = np.zeros((T-lookback,len(assets_to_trade)))
    model.eval() 
 
    with torch.no_grad():  
        # # compute weights sequentially
        # for t in range(lookback,data_test.shape[0]):
        #     idxs_selected = ~np.any(data_test[(t-lookback):t,:] == 0, axis = 0).ravel() 
        #     inputs = np.cumsum(data_test[(t-lookback):t,idxs_selected],axis=0).T #(N,T)
        #     inputs = torch.tensor(inputs,device=device)    
        #     for i in range(len(paths_checkpoints)):  #This ensembles if many checkpoints are given                               
        #         if load_params:
        #             checkpoint = torch.load(paths_checkpoints[i],map_location = device)
        #             model.load_state_dict(checkpoint['model_state_dict'])
        #             model.to(device)              
        #         weightsTest[idxs_selected] += model(inputs).cpu() #tensor.cpu() and tensor.to(torch.device('cpu')) is the same; you cannot transofrm to numpy from gpu
        #     weightsTest /= len(paths_checkpoints)
        #     abs_sum = torch.sum(torch.abs(weightsTest)) #Modify this for quantiles
        #     if abs_sum > 0:
        #         weightsTest /= abs_sum         
        #     retsTest[t-lookback] = torch.sum(weightsTest[idxs_selected]*torch.tensor(data_test[t,idxs_selected],device='cpu')) 
        #     #print(retsTest, retsTest.shape)
            
        weights = torch.zeros((T-lookback,N),device=device)
        for i in range(len(paths_checkpoints)):  #This ensembles if many checkpoints are given                               
            if load_params:
                checkpoint = torch.load(paths_checkpoints[i],map_location = device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
            weights[idxs_selected] += model(torch.tensor(windows[idxs_selected],device=device))
        weights /= len(paths_checkpoints)
        if residual_weights is None:
            abs_sum = torch.sum(torch.abs(weights), axis=1, keepdim=True)
            logging.debug(f"get_returns(): weights abs_sum {abs_sum/len(weights)}")
        else:
            residual_weights = residual_weights[:,assets_to_trade]
            assert(weights.shape == residual_weights[lookback:T,:,0].shape)
            T1,N1 = weights.shape
            weights2 = torch.bmm(weights.reshape(T1,1,N1), torch.tensor(residual_weights[lookback:T],device=device)).squeeze() 
            abs_sum = torch.sum(torch.abs(weights2), axis=1, keepdim=True)
            logging.debug(f"get_returns(): weights2 abs_sum {abs_sum/len(weights2)}")
            # abs_sum_w = torch.sum(torch.abs(weights), axis=1, keepdim=True)
            # logging.info(f"get_returns(): weights2 abs_sum {abs_sum/len(weights2)} weights abs_sum {abs_sum_w/len(weights)}")
            try: weights2 = weights2/abs_sum
            except: weights2 = weights2/(abs_sum + 1e-8)
        try: weights = weights/abs_sum
        except: weights = weights/(abs_sum + 1e-8)
        rets_test = torch.sum(weights * torch.tensor(data_test[lookback:T,:],device=device), axis=1)
        if residual_weights is not None:
            weights = weights2
        turnover = torch.cat((torch.zeros(1,device=device),torch.sum(torch.abs(weights[1:]-weights[:-1]),axis=1)))
        short_proportion = torch.sum(torch.abs(torch.min(weights,torch.zeros(1,device=device))),axis=1)
        rets_test = rets_test-trans_cost*turnover-hold_cost*short_proportion         
        turnover[0] = torch.mean(turnover[1:]) 
        mean = torch.mean(rets_test)
        std =  torch.std(rets_test)
        sharpe = -mean/std
        loss = None
        if objective == "sharpe":
            loss = sharpe
        elif objective == "meanvar":
            loss = -mean*252 + std*15.9
        elif objective == "sqrtMeanSharpe":
            loss = -torch.sign(mean)*torch.sqrt(torch.abs(mean))/std
        else:
            raise Exception(f"Invalid objective loss {objective}")
    return (rets_test.cpu().numpy(), loss, sharpe, turnover.cpu().numpy(), short_proportion.cpu().numpy(), weights.cpu().numpy(), 
           assets_to_trade)


def test(Data, 
         daily_dates,
         model,
         preprocess,
         config,
         residual_weights = None,
         log_dev_progress_freq = 50, log_plot_freq = 199,
         num_epochs = 100, lr = 0.001, batchsize = 150, 
         early_stopping = False,
         save_params = True,
         device = 'cuda',
         output_path = os.path.join(os.getcwd(), 'results', 'Unknown'), model_tag = 'Unknown', 
         lookback = 30, retrain_freq = 250, length_training = 1000, rolling_retrain = True,
         parallelize = True, 
         device_ids=[0,1,2,3,4,5,6,7],  
         trans_cost=0, hold_cost = 0,
         force_retrain = False,
         objective = "sharpe",):
    
    # chooses assets which have at least #lookback non-missing observations in the training period
    assets_to_trade = np.count_nonzero(Data, axis=0) >= lookback
    logging.info(f"test(): assets_to_trade.shape {assets_to_trade.shape}")
    Data = Data[:,assets_to_trade]
    T,N = Data.shape
    returns = np.zeros(T-length_training)
    turnovers = np.zeros(T-length_training)
    short_proportions = np.zeros(T-length_training)
    all_weights = np.zeros((T-length_training, len(assets_to_trade)))
    # load assets_to_trade for weights
    if residual_weights is not None and 'FamaFrenchNew' in model_tag:
        assets_to_trade = np.load('residuals/famafrench-universe/assets-to-consider.npy')
        Data = Data[:,assets_to_trade]
        all_weights = np.zeros((T-length_training,len(assets_to_trade)))
    if residual_weights is not None and 'FamaFrench' in model_tag and 'New' not in model_tag:
        Ndifference = residual_weights.shape[2] - np.sum(assets_to_trade)
        if Ndifference > 0:
            all_weights = np.zeros((T-length_training, len(assets_to_trade) + Ndifference))
            assets_to_trade = np.append(assets_to_trade, np.ones(Ndifference, dtype=np.bool)) 
    if residual_weights is not None and ('IPCA' in model_tag or 'Deep' in model_tag):
        assets_to_trade = np.load('residuals/superMask.npy')
        all_weights = np.zeros((T-length_training, len(assets_to_trade)))
   
    # run train/test over dataset
    for t in range(int( (T-length_training) / retrain_freq ) + 1): 
        logging.info(f'AT SUBPERIOD {t}/{int((T-length_training)/retrain_freq)+1}')
        # logging.info(f"{Data[initialTrain:length_training+(t)*retrain_freq].shape} {Data[length_training+t*retrain_freq:min(length_training+(t+1)*retrain_freq,T)].shape}")            
        data_train_t = Data[t*retrain_freq:length_training+t*retrain_freq]
        data_test_t = Data[length_training+t*retrain_freq-lookback:min(length_training+(t+1)*retrain_freq,T)]
        residual_weights_train_t = None if residual_weights is None \
                                 else residual_weights[t*retrain_freq:length_training+t*retrain_freq]
        residual_weights_test_t = None if residual_weights is None \
                               else residual_weights[length_training+t*retrain_freq-lookback:min(length_training+(t+1)*retrain_freq,T)]
        model_tag_t = model_tag + f'__subperiod{t}'
        
        if rolling_retrain or t == 0:
            model_t = model(logdir=output_path, **config['model'])
            rets_t,turns_t,shorts_t,weights_t,a2t = train(model_t,
                                              preprocess = preprocess,
                                              data_train = data_train_t, 
                                              data_dev = data_test_t,  # dev dataset isn't used as we don't do any validation tuning, so test dataset goes here for progress reporting
                                              residual_weights_train = residual_weights_train_t, 
                                              residual_weights_dev = residual_weights_test_t,  # dev dataset isn't used as we don't do any validation tuning, so test dataset goes here for progress reporting
                                              log_dev_progress_freq = log_dev_progress_freq, 
                                              num_epochs = num_epochs, 
                                              force_retrain = force_retrain,
                                              optimizer_name = config['optimizer_name'],
                                              optimizer_opts = config['optimizer_opts'],
                                              early_stopping = early_stopping,
                                              save_params = save_params, 
                                              output_path = output_path, 
                                              model_tag = model_tag_t,
                                              device = device, 
                                              lookback = lookback, 
                                              log_plot_freq = log_plot_freq, 
                                              parallelize = parallelize, 
                                              device_ids = device_ids, 
                                              batchsize = batchsize,
                                              trans_cost = trans_cost,
                                              hold_cost = hold_cost,
                                              objective = objective,)
            logging.debug("train() completed")
        else:
            rets_t,_,_,turns_t,shorts_t,weights_t,a2t = get_returns(model_t,
                                                        preprocess = preprocess,
                                                        objective = objective,
                                                        data_test = data_test_t,
                                                        residual_weights = residual_weights_test_t,
                                                        device = device, 
                                                        lookback = lookback, 
                                                        trans_cost = trans_cost, 
                                                        hold_cost = hold_cost,)
            logging.debug("get_returns() completed")
             
        returns[t*retrain_freq:min((t+1)*retrain_freq,T-length_training)] = rets_t
        turnovers[t*retrain_freq:min((t+1)*retrain_freq,T-length_training)] = turns_t 
        short_proportions[t*retrain_freq:min((t+1)*retrain_freq,T-length_training)] = shorts_t
        if residual_weights is None:
            w = np.zeros((min((t+1)*retrain_freq,T-length_training) - t*retrain_freq, len(a2t)))
            logging.debug(f"returned weights.shape {weights_t.shape}")
            w[:,a2t] = weights_t
        else:
            w = weights_t
        logging.debug(f"weights selected shape {all_weights[t*retrain_freq:min((t+1)*retrain_freq,T-length_training),assets_to_trade].shape}")
        logging.debug(f"sum(assets_to_trade) {np.sum(assets_to_trade)}")
        all_weights[t*retrain_freq:min((t+1)*retrain_freq,T-length_training),assets_to_trade] = w
        if 'cpu' not in device:
            with torch.cuda.device(device):
                torch.cuda.empty_cache() 
        
    logging.info(f'TRAIN/TEST COMPLETE')
    cumRets = np.cumprod(1+returns)
    plt.figure()
    plt.plot_date(daily_dates[-len(cumRets):], cumRets, marker='None', linestyle='solid')
    plt.savefig(os.path.join(output_path, model_tag + "_cumulative-returns.png"))
    #plt.show()

    plt.figure()
    plt.plot_date(daily_dates[-len(cumRets):], turnovers, marker='None',linestyle='solid')
    plt.savefig(os.path.join(output_path, model_tag + "_turnover.png"))
    #plt.show()
                         
    plt.figure()
    plt.plot_date(daily_dates[-len(cumRets):], short_proportions, marker='None',linestyle='solid')
    plt.savefig(os.path.join(output_path, model_tag + "_short-proportion.png"))
    #plt.show()
    
    np.save(os.path.join(output_path, 'WeightsComplete_' + model_tag + '.npy'), all_weights)
                         
    full_ret = np.mean(returns)
    full_std = np.std(returns)
    full_sharpe = full_ret/full_std
    logging.info(f"==> Sharpe: {full_sharpe*np.sqrt(252) :.2f}, "\
                 f"ret: {full_ret*252 :.4f}, "\
                 f"std: {full_std*np.sqrt(252) :.4f}, "\
                 f"turnover: {np.mean(turnovers) :.4f}, "\
                 f"short_proportion: {np.mean(short_proportions) :.4f}")
                   
    return returns, full_sharpe, full_ret, full_std, turnovers, short_proportions

def estimate(Data, 
             daily_dates,
             model,
             preprocess,
             config,
             residual_weights = None,
             log_dev_progress_freq = 50, log_plot_freq = 199,
             num_epochs = 100, lr = 0.001, batchsize = 150, 
             early_stopping = False,
             save_params = True,
             device = 'cuda',
             output_path = os.path.join(os.getcwd(), 'results', 'Unknown'), model_tag = 'Unknown', 
             lookback = 30, length_training = 1000, test_size=125,
             parallelize = True, 
             device_ids=[0,1,2,3,4,5,6,7],  
             trans_cost=0, hold_cost = 0,
             force_retrain = True,
             objective = "sharpe",
             estimate_start_idx = 0,):
    
    # chooses assets which have at least #lookback non-missing observations in the training period
    assets_to_trade = np.count_nonzero(Data,axis=0) >= lookback 
    Data = Data[:,assets_to_trade]
    T,N = Data.shape
    returns = np.zeros(length_training)
    turnovers = np.zeros(length_training)
    short_proportions = np.zeros(length_training)
    all_weights = np.zeros((length_training,len(assets_to_trade)))
    # load assets_to_trade for weights
    if residual_weights is not None and 'FamaFrenchNew' in model_tag:
        assets_to_trade = np.load('residuals/famafrench-universe/assets-to-consider.npy')
        Data = Data[:,assets_to_trade]
        all_weights= np.zeros((T-length_training,len(assets_to_trade)))
    if residual_weights is not None and 'Fama' in model_tag and 'New' not in model_tag:
        Ndifference = residual_weights.shape[2] - np.sum(assets_to_trade)
        if Ndifference>0:
            all_weights= np.zeros((length_training,len(assets_to_trade)+Ndifference))
            assets_to_trade = np.append(assets_to_trade,np.ones(Ndifference,dtype=np.bool)) 
    if residual_weights is not None and ('IPCA' in model_tag or 'Deep' in model_tag):
        assets_to_trade = np.load('residuals/superMask.npy')
        all_weights= np.zeros((length_training,len(assets_to_trade)))
   
    # estimate over dataset
    logging.info(f"ESTIMATING {estimate_start_idx}:{min(estimate_start_idx+length_training,T)}")
    logging.info(f"TESTING {estimate_start_idx+length_training-lookback}:{min(estimate_start_idx+length_training+test_size,T)}")
    data_train = Data[estimate_start_idx:min(estimate_start_idx+length_training,T)]
    data_dev = Data[estimate_start_idx+length_training-lookback:min(estimate_start_idx+length_training+test_size,T)]
    residual_weights_train = None if residual_weights is None \
                           else residual_weights[estimate_start_idx:min(estimate_start_idx+length_training,T)]
    residual_weights_dev = None if residual_weights is None \
                         else residual_weights[estimate_start_idx+length_training-lookback:min(estimate_start_idx+length_training+test_size,T)]
    del residual_weights
    del Data
    model_tag = model_tag + f'__estimation{estimate_start_idx}-{length_training}-{test_size}'

    model1 = model(logdir=output_path, **config['model'])
    rets,turns,shorts,weights = train(model1,
                                      preprocess = preprocess,
                                      data_train = data_train, 
                                      data_dev = data_dev, 
                                      residual_weights_train = residual_weights_train, 
                                      residual_weights_dev = residual_weights_dev,
                                      log_dev_progress_freq = log_dev_progress_freq, 
                                      num_epochs = num_epochs, 
                                      force_retrain = force_retrain,
                                      lr = lr, 
                                      early_stopping = early_stopping,
                                      save_params = save_params, 
                                      output_path = output_path, 
                                      model_tag = model_tag,
                                      device = device, 
                                      lookback = lookback, 
                                      log_plot_freq = log_plot_freq, 
                                      parallelize = parallelize, 
                                      device_ids = device_ids, 
                                      batchsize = batchsize,
                                      trans_cost = trans_cost,
                                      hold_cost = hold_cost,
                                      objective = objective,)

    returns = rets
    turnovers = turns 
    short_proportions = shorts
    all_weights= weights
    if 'cpu' not in device:
        with torch.cuda.device(device):
            torch.cuda.empty_cache() 
        
    logging.info(f'ESTIMATION COMPLETE')
    
    np.save(os.path.join(output_path, 'WeightsComplete_' + model_tag + '.npy'), all_weights)
                         
    full_ret = np.mean(returns)
    full_std = np.std(returns)
    full_sharpe = full_ret/full_std
    logging.info(f"==> Sharpe: {full_sharpe*np.sqrt(252) :.2f}, "\
                 f"ret: {full_ret*252 :.4f}, "\
                 f"std: {full_std*np.sqrt(252) :.4f}, "\
                 f"turnover: {np.mean(turnovers) :.4f}, "\
                 f"short_proportion: {np.mean(short_proportions) :.4f}")
                   
    return returns, full_sharpe, full_ret, full_std, turnovers, short_proportions