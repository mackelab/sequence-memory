import os,sys
sys.path.append(os.getcwd())
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir+ "/..")
from analysis.summary_parallel import Summary
import numpy as np
base_dir = ""
task_dir = "datasweep_ISI3.pkl"
calc_vex=True
summary_settings = {
    "upsample" : 1, # Increase temporal resolution
    "ref_phase" : "sine", # Reference phase for 'spike-phase' histogram, either sine or LFP
    "onlyGaba" : False,  # Only use inhibitory neurons for calculating LFP
    "cutoff_p": 10e-3, # For Wilc p test
    "normalize_fr_extract":  True,  # Normalize extracted firing rates
    "n_trials": 224,  # Number trials used in analysis
    "randomize_onset": False, # Randomise stimulus onset
    "delay_ms": 2500, # Delay time in ms
    "disable_noise": False, # With or without noise
    "freqs_l": np.logspace(*np.log10([1, 25]), num=50), # Frequencies for spectrograms
    "balance_trials": True, # Draw trials with balanced proportion of each stimuli
    "substr_mean_LFP": False, # Substract mean LFP
    "delay_buffer1": 25, # Disregard short period after stimulus offset
    "delay_buffer2": 25, # Disregard short period before probe onset
    "nbins": 20, # Number of bins for 'spike' phase histograms
    "common_baseline" : True, # Common baseline
    "freqs": [1, 1.5, 1.75, 2.04, 2.37, 2.75, 3.21, 3.73, 4.35, 5], # Frequencies for vex plots
    "ISIs":[5,10]
}

Sum_obj = Summary()


if str(os.popen("hostname").read()) == "Matthijss-MacBook-Air\n":
    data_dir = "../data/"+str(task_dir)
    model_dir = os.path.join(base_dir, "..", "models/sweep_main")
    
else:
    data_dir = (
        "/mnt/qb/work/macke/mpals85/ISI_data/"+str(task_dir)
    )
    model_dir = "/home/macke/mpals85/sequence-memory/rnn_model/models/sweep_main"
data_list, summary_settings = Sum_obj.run_summary(summary_settings, model_dir, data_dir,n_jobs=20, calc_vex=calc_vex)
