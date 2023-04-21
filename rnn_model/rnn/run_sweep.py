SWEEPID = "ydk3it3y"
import sys
import os

from numpy.core.numeric import False_

sys.path.append(os.getcwd())
import numpy as np

# Import utility functions
import wandb
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
# Import the continuous rate model
from model import RNN

# Import the tasks
from task import trial_generator


def sweeper():

    """
    Runs a sweep over models as initialsed by WandB
    """


    #Set up the output dir where the output model will be saved
    # check if on cluster or workstation
    if str(os.popen("hostname").read()) == "MackeLabTaichi\n":
        out_dir = os.path.join(os.getcwd(), "models/sweepWandB")
        gpu_frac = 0.7

    else:
        out_dir = (
            "/mnt/qb/work/macke/mpals85/data"
        )
        gpu_frac = 0.7

    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)

    """
    Define simulation settings
    """


    n_items = 4
    n_channels = 8
    out_channels = 1
    batch_size = 128
    loss = "l2"
    gpu = "0"
    val_perc = 0

    model_params = {
        "n_channels": n_channels,  # n input channels
        "apply_dale": True,  # make network inhibitory-excitory
        "balance_DL": True,  # keep expectation of neuron input 0 after applying Dale's Law
        "rm_outlier_DL": False,  #remove outliers in eigenspectrym by enforcing J 1 = 0
        "spectr_norm": False,  # keep spectral norm to spec_rad by normalisation
        "P_inh": 0.2,  # proportion of inhibitory neurons
        "N": 200,  # n neurons in recurrent layer
        "P_rec": 1,  # probability of being connected in recurrent layer
        "P_in": 1,  # probability of connection between input and recurrent neurons
        "spec_rad": 1.5,  # initial spectral radiance of recurrent layer
        "w_dist": "Gauss",  # w rec distribution, use Gauss or Gamma
        "no_autapses": False,  # no connections to self in recurrent layer
        "out_channels": out_channels,  # number of output channels
        "ex_input": False,  # only positive input conn weights
        "1overN_out": False,  # scale output weights with 1/N (else 1/sqrt(var))
    }

    training_params = {
        # task
        "rand_stim_amp": 0,  # randomise stim amplitude uniformly around 1 with this amount
        "n_items": n_items,  # sequence length
        # onset
        "stim_ons": 125,  # stimulus onset, in timesteps
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
        "delays": [20,260],  # delay lengths (using curr learning)
        "random_delay": 20,  # randomise delay with this amount
        "random_delay_per_tr": True,  # Randomise delay every trial, else every batch
        # training
        "learning_rate": 1e-5,  # learning rate
        "rec_noise": 0.05,  # noise in recurrent layer
        "loss_threshold": 0.8,  # loss threshold (when to stop training)
        "acc_threshold": 0.95,  # accuracy theshold (when to stop training)
        "batch_size": batch_size,  # batch_size
        "eval_freq": 100,  # how often to evaluate task perf
        "eval_tr": int(np.ceil(100 / batch_size)),  # number of trials for eval
        "eval_amp_threh": 0.7,  # amplitude threshold during response window
        "saving_freq": 100000,  # how often to save the model
        "activation": "tanh",  # activation function
        "n_trials": 500000,  # max num of trials
        "clip_max_grad_val": 1,  # max norm for gradient clipping
        "spike_cost": 1e-5,  # l2 reg on firing rate
        "rec_weight_cost": 0,  # 1e-4, # l2 reg on recurrent weights
        "in_weight_cost": 0,  # l2 reg on recurrent weights
        "out_weight_cost": 0,  # l2 reg on output weights
        "lossF": 2.04, # regularisation frequency
        "reg_LFP": True, # regularise LFP (as opposed to single units)
        "osc_cost": 0.5, # oscillatory regularisation amount
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
        "tau_lims": [20, 100],  # 150],   # max and min timeconstant for rnn layer, in ms
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
    name = (
        "SpecRad"
        + str(model_params["spec_rad"])
        + "Dale"
        + str(model_params["apply_dale"])
    )
    net.train(
        training_params,
        model_params,
        trial_gen,
        gpu,
        gpu_frac,
        out_dir,
        name=name,
        sync_wandb=True,
    )


# sweeper()
wandb.agent(SWEEPID, function=sweeper, project="sequence-memory-rnn_model_rnn")