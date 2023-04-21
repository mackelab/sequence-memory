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


# Import the continuous rate model
from model import RNN

# Import the tasks
from task import trial_generator


"""
Set up the output dir where the output model will be saved
"""

# check if on cluster or workstation
#if str(os.popen("hostname").read()) == "MackeLabTaichi\n":
#    out_dir = os.path.join(os.getcwd(), "models")
#else:
#    out_dir = "/home/macke/mpals85/phase-coding/oscillatory_driven/models"
out_dir = "/Users/matthijs/sequence-memory/rnn_model/models/sweep_main"
#out_dir = "/Users/matthijs/sequence-memory/rnn_model/models/new"

if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)

"""
Define simulation settings
"""

gpu = "0"
gpu_frac = 0

net = RNN()


#load existing model

name = "a2r9nqtkSpecRad1.5DaleTrueTC20_100rand0sparse1osc2.75cost0.1"
#name = "SpecRad1.5DaleTrue2.04cost0.1"
net.load_model(os.path.join(out_dir, name))
model_dir = os.path.join(out_dir, name)
var = scipy.io.loadmat(model_dir)
if len(var['val_ind']):
    val_perc = len(var['val_ind'][0])/(len(var['train_ind'][0])+len(var['val_ind'][0]))
else:
    val_perc=0
print("val_perc ", val_perc)
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
training_params['acc_threshold']=0.85
training_params['osc_reg_inh']=False
training_params['random_delay']=20
training_params['random_delay_per_tr']=True
training_params['delays']=[training_params['delays'][-1]+training_params['random_delay']//2]
training_params["learning_rate"]=1e-5

print(training_params)
print(model_params)

net.train(training_params, model_params, trial_gen, gpu, gpu_frac, out_dir, new_run = False,sync_wandb=False)
