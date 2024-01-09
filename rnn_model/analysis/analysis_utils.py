from scipy.stats import wilcoxon
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, os
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import copy
import matplotlib.collections as mcoll
from tf_utils import *
import pickle


def extract_stim_trig_act(
    r1o,
    stim,
    stim_roll,
    settings,
    normalize=False,
    baseline_start=0,
    baseline_len=25,
    stim_len=25,
):
    """
    Get mean baseline and stim activation per stimulus, for every neuron, for every trial

    Args:
        r1o: array of firing rates, shape = [timesteps, n_neurons, n_trials]
        stim: stimuli used to generate r1, shape = [n_channels, n_trials, timesteps]
        stim_roll: duration random onsets, shape = [n_trials]
        settings: settings used to generate stim, Dictionary
        normalize: Whether to Zscore activity with baseline statistics, Bool
        baseline_start: start of baseline period activity, int
        baseline_len: duration of baseline period activity, int
        stim_len: duration of stimulus triggered activity, int

    Returns:
        data: mean activations per neuron per stimulus, [n_channels][n_neurons][2][n_means]
        label: label indices [n_trials]
    """

    n_channels = stim.shape[0]
    n_trials = stim.shape[1]
    r1 = np.copy(r1o)

    N = r1.shape[1]
    # instantiate data array
    data = [[[[], []] for _ in range(N)] for _ in range(n_channels)]

    # get baseline timepoints
    t1 = baseline_start
    t2 = baseline_start + baseline_len
    for trial in range(n_trials):
        if stim_roll[trial] > 0:
            r1[:, :, trial] = np.roll(r1[:, :, trial], -stim_roll[trial], axis=0)
            stim[:, trial] = np.roll(stim[:, trial], -stim_roll[trial], axis=1)
    r1 = r1[settings["rand_ons"] :]
    stim = stim[:, :, settings["rand_ons"] :]
    # normalize
    if normalize:
        r1 = zscore2baseline(r1, axis=0, t1=t1, t2=t2, oscvar=False)

    # get baseline activity
    baseline_act = np.mean(r1[t1:t2], axis=0)

    # get activity for all four presented stimuli
    for it in range(settings["n_items"]):

        # get stimulus activity
        t1 = settings["stim_ons"] + it * (
            settings["stim_dur"] + settings["stim_offs"]
        )  # +min(stim_roll)
        t2 = t1 + stim_len

        stim_act = np.mean(r1[t1:t2], axis=0)

        # get labels
        labels = np.argmax(
            np.sum(stim[:n_channels, :, t1 : t1 + settings["stim_dur"]], axis=2), axis=0
        )

        # run through all labels
        for lab in range(n_channels):
            # find trials with current label
            lab_trials = labels == lab
            # get baseline and stim act of these

            baseline_act_lab = baseline_act[:, lab_trials]
            stim_act_lab = stim_act[:, lab_trials]

            # append the data for all neurons and all trials
            for n in range(N):
                for tr in range(np.sum(lab_trials)):
                    data[lab][n][0].append(baseline_act_lab[n, tr])
                    data[lab][n][1].append(stim_act_lab[n, tr])
    return data, labels


def get_wilc_pvals(data, min_samples=30, onesided=True, common_baseline=True):

    """
    Get wilcoxon pvalues of baseline versus stim pres
    args:
        data: mean activations = [n_channels][n_neurons][2][n_means]
        min_samples: minimum nonzero differences needed for test, else p_val = 1
        onesided: minimum increase in baseline wanted

    returns:
        wilc_pvals: array w pvals, shape:[n_channels, N]

    """

    n_channels = len(data)
    N = len(data[0])
    wilc_pvals = np.ones((n_channels, N))

    # loop through all labels and neurons
    for label in range(n_channels):
        for n in range(N):
            stim_means = data[label][n][1]

            if common_baseline:
                # calc wilc_pval:
                if onesided:
                    wilc_pvals[label, n] = wilcoxon(
                        stim_means, alternative="greater"
                    )[1]
                else:
                    wilc_pvals[label, n] = wilcoxon(stim_means)[1]
            else:      
                baseline_means = data[label][n][0]

                # Check if there is min_sampels amount of non zero differences:
                if (
                    np.count_nonzero((np.array(stim_means) - np.array(baseline_means)))
                    > min_samples
                ):

                    # calc wilc_pval:
                    if onesided:
                        wilc_pvals[label, n] = wilcoxon(
                            stim_means, baseline_means, alternative="greater"
                        )[1]
                    else:
                        wilc_pvals[label, n] = wilcoxon(stim_means, baseline_means)[1]
    return wilc_pvals


def get_dprime(data, wilc_pvals, cutoff=1e-3, div_var=True):
    """
    Calculate d prime for every neuron
    
    Args:
      data: mean activations = [n_channels][n_neurons][2][n_means]
      wilc_pvals: array w pvals, shape:[n_channels, N]
      cutoff: select neurons with pval smaller than this
      div_var: divide dprime mean differences by variances

    Returns:
        d_primes, d prime value for difference in stimulus triggered activity, shape: [N]
        responsive, whether or not a neuron is stimulus responsive, shape:[N]
        prefered_stim, prefered stimulus of each neuron, shape: [N]
    """

    N = len(data[0])

    # for every cell, find the first and second most prefered stim
    pref_stims = np.argsort(wilc_pvals, axis=0)
    prefered_stim1 = pref_stims[0]
    prefered_stim2 = pref_stims[1]

    # calculate d prime value of mean act
    # first versus second pref stim
    responsive = np.zeros(N)
    d_primes = np.zeros(N)
    for n in range(N):
        if wilc_pvals[prefered_stim1[n], n] < cutoff:
            responsive[n] = 1
            means_stim1 = data[prefered_stim1[n]][n]
            means_stim2 = data[prefered_stim2[n]][n]
            d_prime = np.mean(means_stim1) - np.mean(means_stim2)
            if div_var:
                d_prime /= np.mean([np.std(means_stim1), np.std(means_stim2)])
            d_primes[n] = d_prime
            # print(d_prime)
    print(
        "percentage of stim responsive cells: "
        + "{:.2f}".format(np.sum(responsive) / N * 100)
        + str(" %")
    )
    print(
        "percentage of cells d prime > 0: "
        + "{:.2f}".format(np.sum(d_primes > 0) / N * 100)
        + str(" %")
    )

    return d_primes, responsive, prefered_stim1


def extract_traces(
    r1, stim, neuron, pref_stim, settings, extract_LFP=False, var=None, onlyGaba=False
):

    """
    Extract traces of a neuron for its prefered stimulus at different positions
    
    Args:
        r1: array of firing rates [time, neurons, batch]
        stim: array of model input [channels, batches, time]
        neuron: neuron index
        pref_stim: prefered stimulus of this neuron
        settings: dictionary with trial settings
        extract_LFP: whether or not to extract LFPs as well as traces
        var: dictionary with model parameters needed to extract LFP
        onlyGaba: use only inhibitory neurons for computing LFPs
    
    Returns:
        pref_r, list of traces for trials with prefered stimulus [n_items][T, n_trials]
        LFPs: list of local field potentials for these trials [n_items][T, n_trials]


    """

    traces_pref = []
    traces_nonpref = []
    LFPs = []

    if extract_LFP:
        LFP = get_LFP(var, r1, stim, onlyGaba)

    for it in range(settings["n_items"]):

        # extract labels
        t1 = (
            settings["stim_ons"]
            + it * (settings["stim_dur"] + settings["stim_offs"])
            - settings["rand_ons"]
        )
        labels = np.argmax(
            np.sum(stim[:16, :, t1 : t1 + settings["stim_dur"]], axis=2), axis=0
        )

        # use labels to find trials with prefered stim shown
        traces_pref.append(r1[:, neuron, labels == pref_stim])
        non_pref = r1[:, neuron, labels != pref_stim]
        # balance by adding the same amount of non prefs as there are prefs
        traces_nonpref.append(non_pref[:, : sum(labels == pref_stim)])

        if extract_LFP:
            LFPs.append(LFP[:, labels == pref_stim])

    if extract_LFP:
        return traces_pref, traces_nonpref, LFPs
    return traces_pref, traces_nonpref


def draw_spikes(r1, t1, t2, eps=1e-10):
    """
    Extracts spikes during given timeframe
    for one trial
    
    Args: 
        r1: array of firing rates [time, neurons,  n_trials]
        t1: start extracting here
        t2: end extracting here
    Returns:
        all_spikes, shape = [neurons, spiketimes]
    """
    r1_norm = r1  # /(np.max(r1, axis = 0)+eps)
    all_spikes = []
    all_fr = []
    for neuron_ind in range(np.shape(r1)[1]):
        fr = r1_norm[t1:t2, neuron_ind]
        all_fr.append(fr)
        spikes = np.random.uniform(0, 1, t2 - t1) < fr
        all_spikes.append(spikes)
    return all_spikes, all_fr


def get_LFP(var, r1, stim, onlyGaba=False, alpha = 1):
    """
    We use the absolute mean input current to the neurons as LFP
    Args:
        var: dictionary containing relevant model params
        r1: array of firing rates [time, neurons,  n_trials]
        stim: array of model input [channels, batches, time]
        onlyGaba: only use inhibitory neurons
        alpha: dt/tau   
    Returns:
        LFP: [time, trials]
    """
    (T, N, n_trials) = r1.shape
    psc = np.zeros((T, N, n_trials))
    n_inh = np.sum(var["dale_mask"] < 0)
    if onlyGaba:
        for t in range(T):
            ww = np.copy(var["t_w_rec"])
            ww[:, -n_inh:] = 0
            psc[t] = alpha * np.matmul(
                abs(ww), r1[t]
            )  # + np.matmul(abs(var['t_w_in']),stim[:, :, t])
    else:
        for t in range(T):
            psc[t] = alpha * np.matmul(abs(var["t_w_rec"]), r1[t]) +  alpha * np.matmul(
                abs(var["t_w_in"]), stim[:, :, t]
            )
    return np.mean(psc[:, :, :], axis=1)


def zscore2baseline(x, axis, t1=25, t2=100, eps=1e-10, oscvar=True):
    """
    Substract mean and divide by sd of given axis and timeframe
    Args:
        x: numpy array that is to be normalized
        axis: normalise along this axis
        t1: from this index
        t2: to this index
        eps: to avoid divide by 0
        oscvar: Boolean, if so additional sqrt(2) term
    Returns:
        x: normalised array
    """
    baseline = np.take(x, indices=np.arange(t1, t2), axis=axis)
    x -= np.mean(baseline, axis=axis)
    if oscvar:
        x /= np.sqrt(2 * np.var(baseline, axis=axis) + eps)
    else:
        x /= np.std(baseline, axis=axis) + eps
    return x


def avg_power(d_primes, prefered_stim, r1, stim, cwt, f, settings, t1, t2, pad=50):
    """
    Reject neurons based on low power at certain frequency
        Args:     
            d_primes, d prime value for difference in stimulus triggered activity, shape: [N]
            prefered_stim, prefered stimulus of each neuron, shape: [N]
            r1: array of firing rates [time, neurons, batch]
            stim: stimuli used to generate r1, shape = [n_channels, n_trials, timesteps]
            cwt: wavelet
            f: frequency in Hz
            settings: dictionary of trial settings
            t1: start time
            t2: end time
            pad: padding

        Returns:
            neurons_power: power per neuron at specified frequency [N]
    """
    T = r1.shape[0]
    time = np.arange(T) * settings["deltaT"] / 1000

    neurons_power = np.zeros(r1.shape[1])
    for neuron in range(r1.shape[1]):
        pref_stim = prefered_stim[neuron]
        traces_pref, _ = extract_traces(r1, stim, neuron, pref_stim, settings)
        avg_pow = np.zeros(4)
        for stim_pos in range(4):
            power_list = []
            for tr in range(traces_pref[stim_pos].shape[1]):
                r = traces_pref[stim_pos][t1 - pad : t2 + pad, tr]

                _, power = inst_phase(r, cwt, time, f, ref_phase=False)
                if len(power) != (t2 - t1) + 2 * pad:
                    print("Increase timeframe or padding")
                    return None
                power_list.append(np.mean(power[t1:t2]))
            avg_pow[stim_pos] = np.mean(power)
        neurons_power[neuron] = np.mean(avg_pow)

    return neurons_power


def accuracy(
    training_params, r, label, eval_delays, isi_probe, stim_roll=None, cutoff_T=0, verbose=True
):
    """
    Calculate accuracy of model run

        Args:
            training_params: dictionary of training params
            r: array of firing rates [time, n_neurons, batch_size]
            label: label for each trial (match, non-match) [batch_size]
            eval_delays: delays per trial [batch_size]
            isi_probe: how long between probes [batch_size]
            stim_roll: randomised stimulus onset [batch_size]
            cutoff_T: optional cut off time from trial

        Returns:
            accuracy
    """

    batch_size = training_params["batch_size"]
    if stim_roll is None:
        stim_roll = np.zeros(batch_size)
    correct = 0
    probe_time = (
        training_params["stim_ons"]
        + stim_roll
        + training_params["n_items"]
        * (training_params["stim_dur"] + training_params["probe_dur"])
        + eval_delays
        + 1
        + training_params["response_ons"]
        + (training_params["n_items"] - 1) * (training_params["stim_offs"])
        + np.sum(isi_probe, axis=1)
        - cutoff_T
    )
    probe_time = np.int_(probe_time)
    if r.shape[1] == 3:
        for i in range(batch_size):
            if label[i] == 1:
                correct += np.max(
                    r[
                        probe_time[i] : probe_time[i] + training_params["response_dur"],
                        1,
                        i,
                    ],
                    axis=0,
                ) > np.max(
                    r[
                        probe_time[i] : probe_time[i] + training_params["response_dur"],
                        ::2,
                        i,
                    ],
                    axis=(0, 1),
                )
            else:
                correct += np.max(
                    r[
                        probe_time[i] : probe_time[i] + training_params["response_dur"],
                        2,
                        i,
                    ],
                    axis=0,
                ) > np.max(
                    r[
                        probe_time[i] : probe_time[i] + training_params["response_dur"],
                        :2,
                        i,
                    ],
                    axis=(0, 1),
                )
    else:
        for i in range(batch_size):
            sign = np.sign(
                r[probe_time[i] : probe_time[i] + training_params["response_dur"], 0, i]
            )
            if label[i] == 1:
                if np.sum(sign == 1) > np.sum(sign == -1):
                    correct += 1
            else:
                if np.sum(sign == -1) > np.sum(sign == 1):
                    correct += 1
    if verbose:
        print(correct)
        print("accuracy = " + str(correct / batch_size))
    return correct / batch_size


def validation_accuracy(net, settings, var, trial_gen, verbose=True):
    """
    Return accuracy in validation set
    
    Args:
        net: instance of an RNN
        settings: dictionary with task settings
        var: dictionary with model params
        trial_gen: instance of a trial generator

    Returns:
        val_acc: accuracy on validation trials
        train_acc: accuracy on training trials
    """

    if len(var["val_ind"]) == 0:
        print("No validation set")
        return 0, 0

    else:

        n_trials_test = len(var["val_ind"][0])
        settings["batch_size"] = n_trials_test
        trial_gen.val_ind = var["val_ind"][0]
        stim, label, delays, stim_roll, _, isi_probe = trial_gen.generate_input(
            settings, settings["delay"], val=True
        )

        stim = stim.astype(np.float64)
        T = np.shape(stim)[-1]
        z, mask = trial_gen.generate_target(
            settings, label, T, delays, stim_roll, isi_probe
        )
        x1, r1, o1 = net.predict(settings, stim[:, :, :])
        val_acc = accuracy(settings, o1, label, delays, isi_probe, stim_roll)

        stim, label, delays, stim_roll, _, isi_probe = trial_gen.generate_input(
            settings, settings["delay"], val=False
        )

        stim = stim.astype(np.float64)
        z, mask = trial_gen.generate_target(
            settings, label, T, delays, stim_roll, isi_probe
        )
        x1, r1, o1 = net.predict(settings, stim[:, :, :])
        train_acc = accuracy(settings, o1, label, delays, isi_probe, stim_roll)
        if verbose:
            print(
                "Val acc = "
                + str(val_acc * 100)
                + "% , Train acc = "
                + str(train_acc * 100)
                + "%"
            )
        return val_acc, train_acc


def reinstate_params(var):
    """
    Split stored dictionary in model and training parameters
    
    Args:
        var: dictionary of trained model parameters

    Returns:
        training_params: dictionary with parameters used for training
        model_params: dictionary with parameters used to initialize model
    """

    model_params = dict.fromkeys(
        [
            "n_channels",
            "n_osc",
            "apply_dale",
            "P_inh",
            "N",
            "P_rec",
            "P_in",
            "spec_rad",
            "no_autapses",
            "out_channels",
            "ex_input",
        ]
    )

    training_params = dict.fromkeys(
        [
            "stim_ons",
            "stim_dur",
            "stim_offs",
            "stim_jit",
            "stim_jit_dist",
            "delays",
            "probe_offs",
            "probe_dur",
            "probe_jit",
            "probe_jit_dist",
            "response_dur",
            "response_ons",
            "n_items",
            "random_delay",
            "rand_ons",
            "rand_stim_amp",
            "learning_rate",
            "rec_noise",
            "loss_threshold",
            "acc_threshold",
            "batch_size",
            "eval_freq",
            "eval_tr",
            "eval_amp_threh",
            "saving_freq",
            "activation",
            "n_trials",
            "clip_max_grad_val",
            "spike_cost",
            "rec_weight_cost",
            "in_weight_cost",
            "out_weight_cost",
            "lossF",
            "osc_cost",
            "lambda_omega",
            "probe_gain",
            "loss",
            "train_w_in",
            "train_w_in_scale",
            "train_w_rec",
            "train_b_rec",
            "train_w_out",
            "train_b_out",
            "train_taus",
            "train_init_state",
            "rand_init_x",
            "tau_lims",
            "deltaT",

        ]
    )

    for key in var.keys():
        for key_mod in model_params.keys():
            if key_mod == key:
                model_params[key_mod] = np.squeeze(var[key]).tolist()
        for key_train in training_params.keys():
            if key_train == key:
                training_params[key_train] = np.squeeze(var[key]).tolist()

    """
    Make compatible with older models lacking certain param
    """

    if not "stim_jit" in var:
        training_params["stim_jit"] = [0, 1]
        training_params["rand_init_x"] = False
        training_params["probe_jit"] = [0, 1]
        training_params["probe_jit_dist"] = "uniform"
        training_params["probe_jit_dist"] = "uniform"

    if not "rand_init_x" in var and "stim_jit" in var:
        training_params["rand_init_x"] = True
    if not "response_dur" in var:
        training_params["response_dur"] = var["probe_dur"][0][0]
        training_params["response_ons"] = var["probe_ons"][0][0]
        training_params["probe_dur"] = var["stim_dur"][0][0]
        training_params["probe_offs"] = var["stim_offs"][0][0]

    if not "stim_offs" in var:
        training_params["stim_offs"] = 0
    if not "rand_ons" in var:
        training_params["rand_ons"] = 0
    if not "rand_stim_amp" in var:
        training_params["rand_stim_amp"] = 0
    if not "input_phase" in var:
        training_params["input_phase"] = 0
    if not "probe_dur" in var:
        training_params["probe_dur"] = var["stim_dur"][0][0]
    if not "probe_offs" in var:
        training_params["probe_offs"] = var["stim_offs"][0][0]

   

    return model_params, training_params

def upsample_time(upsample, training_params):

        """
        Change timestep
        Args:
            upsample: upsample factor
            training_params: dictionary containing task parameters
        """

        training_params["stim_ons"] *= upsample
        training_params["stim_dur"] *= upsample
        training_params["stim_offs"] *= upsample
        training_params["probe_offs"] *= upsample
        training_params["probe_dur"] *= upsample
        training_params["response_ons"] *= upsample
        training_params["response_dur"] *= upsample
        try:
            training_params["delays"] = [
                delay * upsample for delay in training_params["delays"]
            ]
        except:
            training_params["delays"] *= upsample
        training_params["deltaT"] = int(training_params["deltaT"] / upsample)
        

def steffiscolours():
    """
    Returns colorscheme for Liebe et al. (in preperation) 2022
    
    Returns:
        pltcolors: warm colorscheme
        pltcolors_alt: cold colorscheme
    """

    # red to yellow
    pltcolors = [
        [c / 255 for c in [255, 201, 70, 255]],
        [c / 255 for c in [253, 141, 33, 255]],
        [c / 255 for c in [227, 26, 28, 255]],
        [c / 255 for c in [142, 23, 15, 255]],
    ]
    hexcolors = str([mpl.colors.to_hex(pltcolors[i])[1:] for i in range(4)])

    # green blue
    pltcolors_alt = [
        [c / 255 for c in [161, 218, 180, 255]],
        [c / 255 for c in [65, 182, 196, 255]],
        [c / 255 for c in [34, 94, 168, 255]],
        [c / 255 for c in [10, 30, 69, 255]],
    ]

    a_file = open("matplotlibrc", "r")
    list_of_lines = a_file.readlines()
    list_of_lines[-1] = "axes.prop_cycle: cycler('color'," + hexcolors + ")"
    a_file = open("matplotlibrc", "w")
    a_file.writelines(list_of_lines)
    a_file.close()

    return pltcolors, pltcolors_alt


def extrapolate_delays(t_trials, t_delays, settings, trial_gen, net,verbose=True):
    """
    Accuracy for RNNs for various delay periods
    
    Args:
        t_trials: number of trials to use for testing the RNN
        t_delays: delays used for testing the RNN
        settings: dictionary of trial settings
        trial_gen: instance of a trial generator
        net: instance of an RNN
    Returns:
        accs: Accuracy per trial duration
    
    """
    accs = np.zeros_like(t_delays, dtype=np.float32)

    stims = draw_balanced_trials(n_trials=t_trials//2)
    stim_ind = []
    for i in range(len(stims[0])):
        ind = np.argmax(np.all(np.equal(trial_gen.all_trials_arr,stims[:,i]),axis = 1))
        stim_ind.append(ind)
    trial_ind_match = stim_ind
    trial_ind_non_match = stim_ind
    settings["batch_size"] = t_trials

    for i, delay in enumerate(t_delays):
        if verbose:
            print("delay " + str(delay))
        settings["delay"] = delay
       
        stim, label, delays, stim_roll, isi_stim, isi_probe = trial_gen.generate_input(
            settings,
            settings["delay"],
            val=False,
            stim_ind_match=trial_ind_match,
            stim_ind_non_match=trial_ind_non_match,
        )
        stim = stim.astype(np.float64)
        T = np.shape(stim)[-1]
        settings["T"] = T
        x1, r1, o1= net.predict(settings, stim[:, :, :])
        acc = accuracy(settings, o1, label, delays, isi_probe, stim_roll,verbose=verbose)
        accs[i] = acc

    return accs


def tanh2log(net):
    """
    Swap out tanh for logistic func
    Args:
        net: instance of an RNN
    
    """
    tw_rec = np.copy(net.initializer["w_rec"])
    dale_mask = net.initializer["dale_mask"]
    conn_mask = net.initializer["conn_mask"]
    relu = lambda x: np.maximum(x, 0)
    if net.apply_dale:
        tw_rec = relu(tw_rec)
        tw_rec = np.matmul(tw_rec, dale_mask)
    tw_rec *= conn_mask

    net.initializer["b_rec"] -= tw_rec.dot(np.ones((net.N, 1)))
    net.initializer["w_rec"] = np.copy(net.initializer["w_rec"]) * 2
    net.initializer["w_out"] = np.copy(net.initializer["w_out"]) * 2
    net.initializer["b_out"] -= net.initializer["w_out"].dot(np.ones((net.N, 1)))
    net.activation = "sigmoid"
    
    
def draw_balanced_trials(n_channels=8, n_stim=4,n_trials = 112, max_iter = 10000, verbose =False):
    
    """To do throw warning if convergence is not possible e.g. n_trials/n_channels is not int"""
    n_not_shuff = n_trials
    stims = np.tile(np.arange(n_channels), (n_stim, int(n_trials/n_channels)))
    converged = False

    n_not_shuff_last = n_trials+1
    i=0
    while not converged:


        ind_not_uniq = np.ones(n_trials)
        ind_not_uniq[np.unique(stims, return_index=True, axis=1)[1]]=0
        ind = np.logical_or(stims[0]==stims[1], 
                            np.logical_or(stims[0]==stims[2], 
                                    np.logical_or(stims[0]==stims[3], 
                                          np.logical_or(stims[1]==stims[2], 
                                                np.logical_or(stims[1]==stims[3], 
                                                       np.logical_or(stims[2]==stims[3],ind_not_uniq)
                                                             )
                                                       )
                                                 )
                                         )
                           )
        n_not_shuff = np.sum(ind)
        if verbose:
            print(n_not_shuff)
        if n_not_shuff == 0:
            converged = True
            print("balanced trials converged")
            return stims      

        not_shuff1 = stims[1,ind]
        not_shuff2 = stims[2,ind]
        not_shuff3 = stims[3,ind]

        np.random.shuffle(not_shuff1)
        np.random.shuffle(not_shuff2)
        np.random.shuffle(not_shuff3)

        stims[1,ind]=not_shuff1
        stims[2,ind]=not_shuff2
        stims[3,ind]=not_shuff3

        if n_not_shuff == n_not_shuff_last:
            if verbose:
                print("restarting")
            stims = np.tile(np.arange(n_channels), (n_stim, int(n_trials/n_channels)))
            n_not_shuff_last = n_trials+1
        else:
            n_not_shuff_last = n_not_shuff
        if i == max_iter:
            print("WARNING: balanced trials not converged")

            return stims      
        i+=1



def copy_untrained(net, var):
    """
    Copy untrained network
    """
    untrained_net = copy.deepcopy(net)
    untrained_net.initializer["w_in"] = var['pre_training_state']['w_in'][0][0]
    untrained_net.initializer["w_in_scale"] = var['pre_training_state']['w_in_scale'][0][0]
    untrained_net.initializer["w_rec"] = var['pre_training_state']['w_rec'][0][0]
    untrained_net.initializer["b_rec"] = var['pre_training_state']['b_rec'][0][0]
    untrained_net.initializer["b_out"] = var['pre_training_state']['b_out'][0][0]
    untrained_net.initializer["w_out"] = var['pre_training_state']['w_out'][0][0]
    untrained_net.initializer["taus_gaus"] = var['pre_training_state']['taus_gaus'][0][0]
    return untrained_net



def get_phase_order(freq,isi):
    """
    Get phase order prediction based on frequency and ISI
    """
    data = pickle.load(open("../data/order_pred.pkl",'rb'))    
    freqs = data['freqs']
    isis = data['isis']
    result = data['result']
    i = arg_is_close(freqs,freq)
    j = arg_is_close(isis,isi-200)
    r = int(result[j,i])
    orders = np.array([[3,1,2],[1,3,2],[3,2,1],[2,3,1],[1,2,3],[2,1,3]])
    return r,orders[r]

def arg_is_close(array, value):
    """Returns index of closest value in array"""
    idx = np.argmin(np.abs(array - value))
    return idx

def old_to_new_perm_inds(i):
    """
    utility function between two conventions of permutation indices 
    """
    if i ==0:
        return 1
    elif i ==1:
        return 4
    elif i ==2:
        return 5
    elif i ==3:
        return 2
    elif i ==4:
        return 0
    elif i ==5:
        return 3
    else:
        print("IND should be between 0 and 5")
        
def new_to_old_perm_inds(i):
    """
    utility function between two conventions of permutation indices 
    """
    if i ==0:
        return 4
    elif i ==1:
        return 0
    elif i ==2:
        return 3
    elif i ==3:
        return 5
    elif i ==4:
        return 1
    elif i ==5:
        return 2
    else:
        print("IND should be between 0 and 5")
            