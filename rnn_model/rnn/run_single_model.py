import sys
import os

from numpy.core.numeric import False_

sys.path.append(os.getcwd())
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
if str(os.popen("hostname").read()) == "MackeLabTaichi\n":
    out_dir = os.path.join(os.getcwd(), "models")
else:
    out_dir = "/home/macke/mpals85/phase-coding/oscillatory_driven/models"

if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)

"""
Define simulation settings
"""

# Each time step is 10 ms (changeable)
n_items = 2
n_channels = 2
out_channels = 1
batch_size = 128
loss = "l2"
gpu = "0"
gpu_frac = 0.7
val_perc = 0


model_params = {
    "n_channels": n_channels,  # n input channels
    "apply_dale": False,  # make network inhibitory-excitory
    "balance_DL": True,  # keep expectation of neuron input 0 after applying Dale's Law
    "rm_outlier_DL": True,  # Remove outliers in eigenspectrym by enforcing J 1 = 0
    "spectr_norm": True,  # keep spectral norm to spec_rad by normalisation
    "P_inh": 0.2,  # proportion of inhibitory neurons
    "N": 200,  # n neurons in recurrent layer
    "P_rec": 1,  # probability of being connected in recurrent layer
    "P_in": 1,  # probability of connection between input and recurrent neurons
    "spec_rad": 1.5,  # initial spectral radiance of recurrent layer
    "w_dist": "Gauss",  # w rec distribution, use Gauss or Gamma
    "no_autapses": False,  # no connections to self in recurrent layer
    "out_channels": out_channels,  # number of output channels
    "ex_input": False,  # only positive input conn weights
    "1overN_out": True,  # scale output weights with 1/N (else 1/sqrt(var))
}

training_params = {
    # task
    "rand_stim_amp": 0,  # randomise stim amplitude uniformly around 1 with this amount
    "n_items": n_items,  # sequence length
    # onset
    "stim_ons": 50,  # stimulus onset, in timesteps
    "rand_ons": 0,  # randomise trial onset by this amount
    # stimulus
    "stim_dur": 20,  # stimulus duration
    "stim_offs": 20,  # time between stimuli (expected value incase of dist)
    "stim_jit_dist": "uniform",  # distribution for jittering offsets (uniform, poisson)
    "stim_jit": [0, 1],  #  if unsifrom shift stim by samples from uniform [low, high]
    # probe
    "probe_dur": 20,  # probe duration
    "probe_offs": 20, # time between probes
    "probe_jit_dist": "uniform",  # distribution for jittering offsets
    "probe_jit": [0, 1],  #  if unsifrom shift stim by samples from uniform [low, high]
    # response
    "response_ons": 0,  # response onset
    "response_dur": 40,  # response duration
    # delay
    "delays": [25,150],  # delay lengths (using curr learning)
    "random_delay": 0,  # randomise delay with this amount
    "random_delay_per_tr": True,  # Randomise delay every trial, else every batch

    # training
    "learning_rate": 1e-3,  # learning rate
    "rec_noise": 0.05,  # noise in recurrent layer
    "loss_threshold": 0.05,  # loss threshold (when to stop training)
    "acc_threshold": 0.95,  # accuracy theshold (when to stop training)
    "batch_size": batch_size,  # batch_size
    "eval_freq": 10,  # how often to evaluate task perf
    "eval_tr": int(np.ceil(100 / batch_size)),  # number of trials for eval
    "eval_amp_threh": 0.7,  # amplitude threshold during response window
    "saving_freq": 50,  # how often to save the model
    "activation": "tanh",  # activation function
    "n_trials": 300000,  # max num of trials
    "clip_max_grad_val": 1,  # max norm for gradient clipping
    "spike_cost": 1e-5,  # l2 reg on firing rate
    "rec_weight_cost": 0,  # 1e-4, # l2 reg on recurrent weights
    "in_weight_cost": 0,  # l2 reg on recurrent weights
    "out_weight_cost": 0,  # l2 reg on output weights
    "lossF": 3.5, # regularisation frequency
    "reg_LFP": True, # regularise LFP (as opposed to single units)
    "osc_cost": 0.1, # oscillatory regularisation amount
    "osc_reg_inh": False,  # apply regularisation only to inhibitory neurons
    "probe_gain": 1,  # put emphasis on decision period
    "loss": loss,  # which loss function to use (sce or l2)
    "rand_init_x": True, #randomise x0 every trial
    
    # variables to learn
    "train_w_in": True,  # input weights
    "train_w_in_scale": False,  # scalar mutliplying input weights
    "train_w_rec": True,  # recurrent weights
    "train_b_rec": False,  # train recurrent bias
    "train_w_out": True,  #  train output weights
    "train_b_out": False,  # train output bias
    "train_taus": True,  # train time constants
    "train_init_state": False,  # train initial state

    # synapse params
    "tau_lims": [100],  # 150],   # max and min timeconstant for rnn layer, in ms
    "deltaT": 10,  # simulation timestep, in ms

}


net = RNN()


"""
initialize model and start training
"""

net.initialize_model(model_params)
print("initialized model")

trial_gen = trial_generator(
    n_items,
    n_channels,
    out_channels,
    val_perc,
)
print("initialized trial gen")
net.train(
    training_params,
    model_params,
    trial_gen,
    gpu,
    gpu_frac,
    out_dir,
    sync_wandb=True,
)


"""
#load existing model

name = "N_items_4_N_200_Delay_125_Acc_0.953125_tr3101_2021_09_14_175143.mat"
net.load_model(os.path.join(out_dir, name))
trial_gen = trial_generator(n_items, n_channels, out_channels, n_osc,  \
        rand_phase_wit, rand_phase_bet, freq_range, val_perc)
model_dir = os.path.join(out_dir, name)
var = scipy.io.loadmat(model_dir)
#trial_gen.val_ind = var['val_ind'][0]
trial_gen.train_ind = var['train_ind'][0]
#net.initializer['w_in_scale']=1.0
net.initializer['w_rec']*=2
net.activation='sigmoid'
print("loaded model")
net.train(training_params, model_params, trial_gen, gpu, gpu_frac, out_dir, new_run = False)
"""
