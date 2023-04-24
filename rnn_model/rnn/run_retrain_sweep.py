SWEEPID = "lccht36k"
import sys
import os

from numpy.core.numeric import False_

sys.path.append(os.getcwd())
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir+ "/..")
from analysis.analysis_utils import *

import time
import scipy.io
import numpy as np
import argparse
import datetime
import wandb

# Import the continuous rate model
from model import RNN

# Import the tasks
from task import trial_generator



def sweeper():
    wandb.init(
        project="phase-coding",
        group="osc_driven_sweep",
        #config={**model_params, **training_params},
    )  # , reinit=True)
    """
    Runs a sweep over models as initialsed by WandB
    """


    #Set up the output dir where the output model will be saved
    # check if on cluster or workstation
    if str(os.popen("hostname").read()) == "Matthijss-MacBook-Air\n":
        out_dir = "/Users/matthijs/sequence-memory/rnn_model/models/sweep_main"
        gpu_frac = 0

    else:
        out_dir = (
            "/mnt/qb/work/macke/mpals85/retrain"
        )
        model_dir = "/home/macke/mpals85/sequence-memory/rnn_model/models/sweep_main"
        gpu_frac = 0.7

    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)

    """
    Define simulation settings
    """

    gpu = "0"
    gpu_frac = .7

    net = RNN()


    #load existing model

    name = wandb.config.model_name
    print(name)
    #name = "SpecRad1.5DaleTrue2.04cost0.1"
    net.load_model(os.path.join(model_dir, name))
    model_dir = os.path.join(model_dir, name)
    var = scipy.io.loadmat(model_dir)
    if len(var['val_ind']):
        val_perc = len(var['val_ind'][0])/(len(var['train_ind'][0])+len(var['val_ind'][0]))
    else:
        val_perc=0
    model_params, training_params = reinstate_params(var)
    out_channels = net.out_channels
    n_channels = net.n_channels
    n_items = int(var["n_items"][0][0])
    trial_gen = trial_generator(n_items, n_channels, out_channels,val_perc)
    if len(var['val_ind']):
        trial_gen.val_ind = var['val_ind'][0]
    else:
        trial_gen.val_ind = []

    trial_gen.train_ind = var['train_ind'][0]

    print("loaded model")
    training_params['acc_threshold']=0.95
    training_params['osc_reg_inh']=False
    training_params['random_delay']=20
    training_params['random_delay_per_tr']=True
    training_params['delays']=[training_params['delays'][-1]+training_params['random_delay']//2]
    training_params["learning_rate"]=1e-5
    training_params["n_trials"]=50000
    training_params['osc_cost']=.5

    wandb.config.update({**model_params, **training_params})
    net.train(training_params, model_params, trial_gen, gpu, gpu_frac, out_dir, new_run = False,sync_wandb=True)



# sweeper()
wandb.agent(SWEEPID, function=sweeper, project="sequence-memory-rnn_model_rnn")
