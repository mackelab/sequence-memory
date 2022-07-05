import scipy.io
import os, sys
sys.path.insert(0,'..')
from rnn.model import RNN
from rnn.task import trial_generator
import numpy as np
from tf_utils import *
from analysis.analysis_utils import *
from scipy.stats import zscore
from pycircstat.tests import watson_williams_test
from pycircstat.distributions import kappa
from itertools import permutations
import pickle

def run_summary(summary_settings, model_dir, data_dir):
    
    """
    Calculates summary statistics for many models

    Args:
        summary_settings: dictionary of settings to use
        model_dir: String denoting folder with models
        data_dir: String denoting folder to store the result of this function

    Returns:
        data_list: Dictionary with summary statistics
        summary_settings: dictionary of used settings
    
    """

    data_list = {
    "model_names":[],
    "loss_f": [],
    "acc":[],
    "val_acc":[],
    "train_acc":[],
    "pre_spectrum":[],
    "post_spectrum":[],
    "ranked_neurons":[],
    "wilc_ps":[],
    "d_primes":[],
    "vex":[],
    "shvex":[],
    "vex_f":[],
    "phase_order":[],
    "kappas":[],
    "low_vex" :[],
    "low_shvex":[],
    "low_kappas":[]
    }

    
    data_list["summary_settings"]=summary_settings
    
    files_and_directories = os.listdir(model_dir)
    for fi, file in enumerate(files_and_directories):
        if file[0]=='.':
            print("removing: " +str(file))
            files_and_directories.pop(fi)

    if os.path.isfile(data_dir):
        with open(data_dir, "rb") as fp:   #Pickling
            data_list = pickle.load(fp)
        return data_list, summary_settings

    for findex, fname in enumerate(files_and_directories):
        print(fname)
        
        """Load Model"""
        data_list["model_names"].append(fname)     
        model_file = os.path.join(model_dir, fname)
        net = RNN()
        var = scipy.io.loadmat(model_file)
        #print(var)
        print("FREQUENCY = " + str(var['lossF'][0][0]))
        data_list["loss_f"].append(var['lossF'][0][0])
        net.load_model(model_file)
        if summary_settings["disable_noise"]:
            net.rec_noise = 0.


        """Instantiate Variables"""
        out_channels = net.out_channels
        n_channels = net.n_channels
        n_items = int(var["n_items"][0][0])
        n_osc = var["w_in"].shape[1] - n_channels
        N = var["N"][0][0]

        model_par, settings = reinstate_params(var)
        delay = int(summary_settings["delay_ms"]/settings['deltaT'])
        settings["delay"] = delay *summary_settings["upsample"]
        settings["stim_ons"] = 125
        if summary_settings["randomize_onset"]:
            settings["rand_ons"] = int(1000/(var['lossF'][0][0]*settings['deltaT']))
        if settings["rand_ons"]>settings["stim_ons"]:
            print("WARNING, can't use this random onset, defaulting to 50 steps")
            settings["rand_ons"]=50
        settings["random_delay"] = 0
        settings["random_delay_per_tr"] = True
        settings["stim_jit_dist"] = "uniform"
        settings["stim_jit"] = [0, 1]
        settings["probe_jit_dist"] = "uniform"
        settings["probe_jit"] = [0, 1]
        settings["batch_size"] = summary_settings["n_trials"]        
        n_trials = summary_settings["n_trials"]  
        upsample_time(summary_settings["upsample"], settings)



        delay_start = (
            settings["stim_ons"]
            + n_items * settings["stim_dur"]
            + (n_items - 1) * settings["stim_offs"]
        )
        delay_end = delay_start + settings["delay"]

        # amount of timesteps per second
        dt_sec = int(1000 / settings["deltaT"])

        # timestep in seconds
        timestep = settings["deltaT"] / 1000



        """
        Trial generator
        """
        val_perc = 0
        rand_phase_bet = "none"
        rand_phase_wit = True  #
        freq_range = [1, 5]
        trial_gen = trial_generator(
            n_items,
            n_channels,
            out_channels,
            val_perc,
        )

        trial_gen.train_ind = var["train_ind"][0]
        trial_gen.phase = var["phase"]
        trial_gen.freq = var["freq"]
    
        "Do val test"
        val_acc, train_acc = validation_accuracy(net, settings, var, trial_gen)
        data_list["val_acc"].append(val_acc)
        data_list["train_acc"].append(train_acc)
        settings["batch_size"] = summary_settings["n_trials"]        

        
        """Generate trials + accuracy"""
        if summary_settings["balance_trials"]:
            stims = draw_balanced_trials()
            stim_ind = []
            for i in range(len(stims[0])):
                ind = np.argmax(np.all(np.equal(trial_gen.all_trials_arr,stims[:,i]),axis = 1))
                stim_ind.append(ind)
            trial_ind_match = stim_ind
            trial_ind_non_match = stim_ind
        else:
            try:
                trial_ind = np.random.choice(
                    np.arange(len(trial_gen.train_ind)), n_trials, replace=False
                )
            except:
                print("WARNING!! : more test trials then possible unique trials selected")
                trial_ind = np.random.choice(
                    np.arange(len(trial_gen.train_ind)), n_trials, replace=True
                )
            label = np.random.choice([0, 1], n_trials)
            trial_ind_match = trial_ind[label == 1]
            trial_ind_non_match = trial_ind[label == 0]
        stim, label, delays, stim_roll, isi_stim, isi_probe = trial_gen.generate_input(
            settings,
            settings["delay"],
            val=False,
            stim_ind_match=trial_ind_match,
            stim_ind_non_match=trial_ind_non_match,
        )

        stim = stim.astype(np.float64)
        T = np.shape(stim)[-1]
        z, mask = trial_gen.generate_target(settings, label, T, delays, stim_roll, isi_probe)
        settings["T"] = T
        time = np.arange(T) * settings["deltaT"] / 1000
        plt_time = (
            np.arange(-settings["stim_ons"] + settings["rand_ons"], T - settings["stim_ons"])
            * settings["deltaT"]
            / 1000
        )

        untrained_net = copy_untrained(net, var)
        x1, r1, o1  = net.predict(settings, stim[:, :, :])
        xu, ru, ou= untrained_net.predict(settings, stim[:, :, :])        
        acc = accuracy(settings, o1, label, delays, isi_probe, stim_roll)
        data_list["acc"].append(acc)

        r1+=1
        r1/=2
        ru+=1
        ru/=2

        "Extract LFP"
        LFP = get_LFP(var, r1, stim, onlyGaba=summary_settings["onlyGaba"])
        LFP = zscore(LFP, axis=0)
        LFP_u = get_LFP(var, ru, stim, onlyGaba=summary_settings["onlyGaba"])
        LFP_u = zscore(LFP_u, axis=0)


        if summary_settings["substr_mean_LFP"]:
            substr = np.mean(LFP, axis=1)
            substr_u = np.mean(LFP_u, axis=1)

        else:
            substr = 0
            substr_u = 0


        amps = []
        amps_u = []

        for tr in range(n_trials):
            _, amp = scalogram(
                LFP[:, tr] - substr,
                7,
                time,
                settings["deltaT"] / 1000,
                summary_settings["freqs_l"],
            )
            _, amp_u = scalogram(
                LFP_u[:, tr] - substr_u,
                7,
                time,
                settings["deltaT"] / 1000,
                summary_settings["freqs_l"],
            )
            amps.append(amp)
            amps_u.append(amp_u)

        amp = np.mean(np.array(amps),axis=0)
        amp_u = np.mean(np.array(amps_u),axis=0)
        
        data_list["post_spectrum"].append(amp)
        data_list["pre_spectrum"].append(amp_u)
        
        main_freq = summary_settings["freqs_l"][np.argmax(np.mean(amp[:, delay_start:delay_end], axis=1))]
        main_power = np.max(np.mean(amp[:, delay_start:delay_end], axis=1))
        baseline_freq = summary_settings["freqs_l"][np.argmax(np.mean(amp[:, :settings["stim_ons"]], axis=1))]
        baseline_power = np.max(np.mean(amp[:, :settings["stim_ons"]], axis=1))

        print("delay freq = " + str(main_freq) + " with power " + str(main_power) +
              "\nbaseline freq = " + str(baseline_freq) + " with power " + str(baseline_power))

        """Stim Trig Activity"""
        baseline_len = int(1000/(baseline_freq*settings['deltaT']))
        baseline_start = max((settings["stim_ons"]-settings['rand_ons'] - baseline_len) // 2, 0)
        stim_len = min(settings["stim_dur"] + settings["stim_offs"], baseline_len)

        # extract stimulus triggered activity
        data, labels = extract_stim_trig_act(
            r1,
            stim,
            stim_roll,
            settings,
            baseline_start=baseline_start,
            baseline_len=baseline_len,
            stim_len=stim_len,
            normalize=summary_settings["normalize_fr_extract"],
        )

        wilc_pvals = get_wilc_pvals(data, onesided=True, common_baseline=True)
        d_primes, responsive, prefered_stim = get_dprime(data, wilc_pvals, cutoff=summary_settings["cutoff_p"])
        ranked_neurons = np.argsort(d_primes)


        data_list["ranked_neurons"].append(ranked_neurons)
        data_list["d_primes"].append(d_primes)
        data_list["wilc_ps"].append(wilc_pvals)
        cutoff_d = np.median(d_primes[responsive.astype(bool)])
        print("Median d prime: " +str(cutoff_d))
        print(
            "percentage of cells d prime > cutoff: "
            + str(np.sum(d_primes > cutoff_d) / N * 100)
        )

        up50th = np.arange(200)[np.logical_and(d_primes>cutoff_d, responsive)]
        low50th = np.arange(200)[np.logical_and(d_primes<cutoff_d, responsive)]


        """
        Calculate VEX

        """
        # freqs = np.arange(main_freq-2/3, main_freq+2/3, 1/3)#3.8
        # freqs = np.arange(0.25, 1.5, 0.25)#3.8

        t1 = delay_start + summary_settings["delay_buffer1"] - settings["rand_ons"]
        t2 = delay_end - summary_settings["delay_buffer2"] - settings["rand_ons"]
        delay_time = time[t1:t2]
        for f in summary_settings["freqs"]:
            if dt_sec / f < summary_settings["nbins"]:
                print("Warning: too much bins for f = " + str(f))

        bin_lims = np.linspace(-np.pi, np.pi, summary_settings["nbins"] + 1)
        bin_centers = bin_lims[:-1] + np.pi / summary_settings["nbins"]
        width = 2 * np.pi / (summary_settings["nbins"])

        vex = np.zeros((len(summary_settings["freqs"]), N))
        kappas = np.zeros((len(summary_settings["freqs"]), N))
        shvex = np.zeros((len(summary_settings["freqs"]), N))
        shuffle_ind = np.random.choice(np.arange(n_items), n_trials)

        for neui, neuron in enumerate(up50th):

            if neui % 10 == 0:
                print("{:.2f}% done".format(100 * neui / len(up50th)))

            pref_stim = prefered_stim[neuron]
            pref_r, _, LFPs = extract_traces(
                r1, stim, neuron, pref_stim, settings, True, var, summary_settings["onlyGaba"]
            )
            for fi, f in enumerate(summary_settings["freqs"]):

                watsdat = []
                watsw = []
                watsw_shuffle = []
                spikephasehist_shuffle = np.zeros((4,summary_settings["nbins"]))
                counter = 0
                cwt = complex_wavelet(timestep, f, 7)
                kappa_n = np.zeros(4)
                for stim_pos in range(settings["n_items"]):
                    spikephasehist = np.zeros(summary_settings["nbins"])
                    for tr in range(np.array(LFPs[stim_pos]).shape[1]):
                        if summary_settings["ref_phase"] == "sine":
                            LFP_phase = wrap(time * 2 * np.pi * f)
                        elif summary_settings["ref_phase"] == "LFP":
                            LFP_phase, _ = inst_phase(
                                LFPs[stim_pos][:, tr], cwt, time, f, ref_phase=False
                            )
                        else:
                            print("WARNING: reference phase not recognised!")

                        bin_ind = np.digitize(LFP_phase[t1:t2], bin_lims) - 1
                        firing_trace = pref_r[stim_pos][t1:t2, tr]
                        # firing_trace -= min(firing_trace)
                        # firing_trace  /= max(firing_trace)+1e-5
                        for b in range(summary_settings["nbins"]):
                            summed_spikes = np.sum(firing_trace[bin_ind == b])
                            occ = np.count_nonzero(bin_ind == b)
                            if occ > 0:
                                spikephasehist[b] += summed_spikes / occ
                                spikephasehist_shuffle[shuffle_ind[counter], b] += (
                                    summed_spikes / occ
                                )
                        counter += 1

                    avg, avglen = circ_mean(bin_centers, spikephasehist)
                    watsw.append(np.array(spikephasehist))
                    watsdat.append(bin_centers)
                    kappa_n[stim_pos] = kappa(bin_centers, w = spikephasehist)
                for stim_pos in range(settings["n_items"]):
                    watsw_shuffle.append(np.array(spikephasehist_shuffle[stim_pos]))

                anovatable = watson_williams_test(
                    bin_centers, bin_centers, bin_centers, bin_centers, w=watsw_shuffle
                )[1]
                shvex[fi, neuron] = anovatable["SS"][0] / anovatable["SS"][2]

                anovatable = watson_williams_test(
                    watsdat[0], watsdat[1], watsdat[2], watsdat[3], w=watsw
                )[1]
                vex[fi, neuron] = anovatable["SS"][0] / anovatable["SS"][2]
                kappas[fi, neuron] = np.mean(kappa_n)
        vex_fr_ind = np.argmax(np.sum(vex, axis=1))
        #vex_fr_ind = np.where(np.isclose(freqs,var['lossF'][0][0]))[0][0]#np.argmax(np.sum(vex, axis=1))
        vex_fr = summary_settings["freqs"][vex_fr_ind]
        print("highest vex at fr: {:.2f}".format(vex_fr))
        data_list["vex"].append(vex)
        data_list["shvex"].append(shvex)
        data_list["vex_f"].append(vex_fr)
        data_list["kappas"].append(kappas)


        """
        calculate vex low 50th
        """

        low_vex = np.zeros((len(summary_settings["freqs"]), N))
        low_kappas = np.zeros((len(summary_settings["freqs"]), N))
        low_shvex = np.zeros((len(summary_settings["freqs"]), N))

        for neui, neuron in enumerate(low50th):

            if neui % 10 == 0:
                print("{:.2f}% done".format(100 * neui / len(low50th)))

            pref_stim = prefered_stim[neuron]
            pref_r, _, LFPs = extract_traces(
                r1, stim, neuron, pref_stim, settings, True, var, summary_settings["onlyGaba"]
            )
            for fi, f in enumerate(summary_settings["freqs"]):

                watsdat = []
                watsw = []
                watsw_shuffle = []
                spikephasehist_shuffle = np.zeros((4,summary_settings["nbins"]))
                counter = 0
                cwt = complex_wavelet(timestep, f, 7)
                kappa_n = np.zeros(4)
                for stim_pos in range(settings["n_items"]):
                    spikephasehist = np.zeros(summary_settings["nbins"])
                    for tr in range(np.array(LFPs[stim_pos]).shape[1]):
                        if summary_settings["ref_phase"] == "sine":
                            LFP_phase = wrap(time * 2 * np.pi * f)
                        elif summary_settings["ref_phase"] == "LFP":
                            LFP_phase, _ = inst_phase(
                                LFPs[stim_pos][:, tr], cwt, time, f, ref_phase=False
                            )
                        else:
                            print("WARNING: reference phase not recognised!")

                        bin_ind = np.digitize(LFP_phase[t1:t2], bin_lims) - 1
                        firing_trace = pref_r[stim_pos][t1:t2, tr]
                        # firing_trace -= min(firing_trace)
                        # firing_trace  /= max(firing_trace)+1e-5
                        for b in range(summary_settings["nbins"]):
                            summed_spikes = np.sum(firing_trace[bin_ind == b])
                            occ = np.count_nonzero(bin_ind == b)
                            if occ > 0:
                                spikephasehist[b] += summed_spikes / occ
                                spikephasehist_shuffle[shuffle_ind[counter], b] += (
                                    summed_spikes / occ
                                )
                        counter += 1

                    avg, avglen = circ_mean(bin_centers, spikephasehist)
                    watsw.append(np.array(spikephasehist))
                    watsdat.append(bin_centers)
                    kappa_n[stim_pos] = kappa(bin_centers, w = spikephasehist)
                for stim_pos in range(settings["n_items"]):
                    watsw_shuffle.append(np.array(spikephasehist_shuffle[stim_pos]))

                anovatable = watson_williams_test(
                    bin_centers, bin_centers, bin_centers, bin_centers, w=watsw_shuffle
                )[1]
                low_shvex[fi, neuron] = anovatable["SS"][0] / anovatable["SS"][2]

                anovatable = watson_williams_test(
                    watsdat[0], watsdat[1], watsdat[2], watsdat[3], w=watsw
                )[1]
                low_vex[fi, neuron] = anovatable["SS"][0] / anovatable["SS"][2]
                low_kappas[fi, neuron] = np.mean(kappa_n)

        data_list["low_vex"].append(low_vex)
        data_list["low_shvex"].append(low_shvex)
        data_list["low_kappas"].append(low_kappas)

        """
        Phase order
        """
        # SUM UP ALL HISTOGRAMS
        # AND CALCULATE PERCENTAGE MATCHING ORDER
        f = vex_fr
  
        bin_lims = np.linspace(-np.pi, np.pi, summary_settings["nbins"] + 1)
        bin_centers = bin_lims[:-1] + np.pi / summary_settings["nbins"]
        width = 2 * np.pi / (summary_settings["nbins"])


        # Initialize histograms
        totalhist_stim = np.zeros((settings["n_items"],summary_settings["nbins"]))
        totalhist_phase = np.zeros((settings["n_items"], summary_settings["nbins"]))

        # For counting Percentage matching order
        perms = list(set(permutations([1, 2, 3])))
        n_match = np.zeros(len(perms))

        # Loop through all neurons to be included
        for neui, neuron in enumerate(up50th):

            
            order = np.zeros(3)
            avgs = np.zeros(4)
            neuronhist = np.zeros((settings["n_items"], summary_settings["nbins"]))

            # Extract trials for prefered stimulus
            pref_stim = prefered_stim[neuron]
            pref_r, _, LFPs = extract_traces(
                r1, stim, neuron, pref_stim, settings, True, var, summary_settings["onlyGaba"]
            )

            # Calculate hist per stim position
            for stim_pos in range(settings["n_items"]):
                spikephasehist = np.zeros(summary_settings["nbins"])

                # Calculate hist per trial
                for tr in range(np.array(LFPs[stim_pos]).shape[1]):

                    if summary_settings["ref_phase"] == "sine":
                        LFP_phase = wrap(time * 2 * np.pi * f)
                    elif summary_settings["ref_phase"] == "LFP":
                        LFP_phase, _ = inst_phase(
                            LFPs[stim_pos][:, tr], cwt, time, f, ref_phase=False
                        )
                    else:
                        print("WARNING: reference phase not recognised!")
                    bin_ind = np.digitize(LFP_phase[t1:t2], bin_lims) - 1
                    firing_trace = pref_r[stim_pos][t1:t2, tr]
                    firing_trace -= min(firing_trace)
                    firing_trace /= max(firing_trace) + 1e-5
                    for b in range(summary_settings["nbins"]):
                        summed_spikes = np.sum(firing_trace[bin_ind == b])

                        # Normalize by bin occurance
                        occ = np.count_nonzero(bin_ind == b)
                        if occ > 0:
                            spikephasehist[b] += summed_spikes / occ

                avgs[stim_pos] = circ_mean(bin_centers, np.array(spikephasehist))[0]
                neuronhist[stim_pos] = np.array(spikephasehist)

            # Calculate order of phases
            phase_order = np.argsort(avgs)

            # Calculate amount matching certain stim order
            avgs -= avgs[0]
            avgs[avgs < 0] += np.pi * 2
            for permi, perm in enumerate(perms):
                if (np.argsort(avgs)[1:] == np.array(perm)).all():
                    n_match[permi] += 1
                    #print("neuron no: " + str(neuron) + "phase order " + str(permi))

        #print("appending phase order:" + str(n_match))
        data_list["phase_order"].append(n_match)


    with open(data_dir, "wb") as fp:   #Pickling
        pickle.dump(data_list, fp)
    return data_list, summary_settings