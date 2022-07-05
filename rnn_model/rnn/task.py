import numpy as np
from itertools import permutations


class trial_generator:
    def __init__(
        self,
        n_items,
        n_channels,
        out_channels,
        val_perc,
    ):

        """
        This class generates the multi-item delayed match to sample trials

        Args:
            n_items: number of items shown per trial
            n_channels: number of input channels
            out_channels: number of output channels
            val_perc: validation percentage, not used during training

        """
        self.n_items = n_items
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.val_perc = val_perc

        # generate an array listing all posible_trials
        self.all_trials_arr = self.generate_all_trials()
        n_trials = self.all_trials_arr.shape[0]

        # generate indices of validation and train trials
        # because we technically don't need the last item to succesfully solve the task
        # we have to make sure the validation set is truly unique

        self.val_ind = []
        self.train_ind = []

        n_equal = n_channels - n_items + 1
        unique = int(n_trials / n_equal)
        n_val_unique = int(val_perc * unique)
        all_trials_ind = np.arange(unique)
        np.random.shuffle(all_trials_ind)
        val_ind_unique = all_trials_ind[:n_val_unique]
        train_ind_unique = all_trials_ind[n_val_unique:]

        ind = 0
        for i in val_ind_unique:
            for j in range(n_equal):
                self.val_ind.append(i * n_equal + j)
                ind += 1

        ind = 0
        for i in train_ind_unique:
            for j in range(n_equal):
                self.train_ind.append(i * n_equal + j)
                ind += 1

        self.train_ind = np.array(self.train_ind)
        self.val_ind = np.array(self.val_ind)

     

    def generate_all_trials(self):
        """
        generates list of all possible trials
        """
        all_trials = set(permutations(np.arange(self.n_channels), int(self.n_items)))
        return np.array(list(all_trials))

    def gen_match_trials(self, n_trials, trial_ind):
        """
        generate all match trials
        """
        if self.deterministic:
            return self.all_trials_arr[trial_ind]
        else:
            rand_ind = np.random.choice(trial_ind, n_trials)
            return self.all_trials_arr[rand_ind]

    def gen_non_match_trials(self, n_trials, trial_ind):
        """
        generate all non-match trials
        """
        if self.deterministic:
            rand_ind = trial_ind
        else:
            rand_ind = np.random.choice(trial_ind, n_trials)

        stims = self.all_trials_arr[rand_ind]
        if self.n_items > 1:
            probes = np.copy(stims)
            match = np.ones(n_trials, dtype=bool)
            matched = True
            # generate probes that do not match
            while matched:
                # indices of trials that match
                to_shuffle = probes[match]
                # shuffle all matching trials
                [np.random.shuffle(x) for x in to_shuffle]
                probes[match] = to_shuffle
                # check if there are still matching trials left
                match = (stims == probes).all(axis=1)
                if np.sum(match) == 0:
                    matched = False
        else:
            probes = (np.copy(stims) - 1) * -1
        return stims, probes

    def generate_input(
        self, settings, delay, val, stim_ind_match=None, stim_ind_non_match=None
    ):
        """
        Generate a set of inputs (evenly distr between match and non-match)
       
        Args:
            settings: dictionary of task parameters
            delay: delay in time steps
            val: Boolean indicating whether validation or training trials
            stim_ind_match: match ttrial indices (optional)
            stim_ind_non_match: non match trial indices (optional)

        Returns:
            u: task input
            labels: associated labels (match or non match)
            delays_u: delays per trial
            stim_roll: randomised onset of first stim
            isi_stim: interstimulus intervals stim period
            isi_probe: interstimulus intervals probe period
        """

        stim_ons = settings["stim_ons"]
        rand_ons = settings["rand_ons"]

        stim_dur = settings["stim_dur"]
        stim_offs = settings["stim_offs"]
        stim_jit_dist = settings["stim_jit_dist"]
        stim_jit = settings["stim_jit"]

        probe_dur = settings["probe_dur"]
        probe_offs = settings["probe_offs"]
        probe_jit_dist = settings["probe_jit_dist"]
        probe_jit = settings["probe_jit"]

        response_ons = settings["response_ons"]
        response_dur = settings["response_dur"]

        n_items = settings["n_items"]

        self.deterministic = False

        if stim_ind_match is None and stim_ind_non_match is None:
            """Randomly sample trials"""
            batch_size = settings["batch_size"]
            if val:
                trial_ind_match = self.val_ind
                trial_ind_non_match = self.val_ind

            else:
                trial_ind_match = self.train_ind
                trial_ind_non_match = self.train_ind
            labels = np.random.randint(2, size=batch_size)

        elif stim_ind_match is None and stim_ind_non_match is not None:
            """deterministicly take non match trials"""
            self.deterministic = True
            batch_size = len(stim_ind_non_match)
            trial_ind_match = []
            trial_ind_non_match = stim_ind_non_match
            labels = np.zeros(batch_size, dtype=int)

        elif stim_ind_match is not None and stim_ind_non_match is None:
            """deterministicly take match trials"""
            self.deterministic = True
            batch_size = len(stim_ind_match)
            trial_ind_match = stim_ind_match
            trial_ind_non_match = []
            labels = np.ones(batch_size, dtype=int)

        else:
            """deterministicly take match and non match trials"""
            self.deterministic = True
            batch_size = len(stim_ind_non_match) + len(stim_ind_match)
            trial_ind_match = stim_ind_match
            trial_ind_non_match = stim_ind_non_match
            labels = np.concatenate(
                [
                    np.zeros(len(stim_ind_non_match), dtype=int),
                    np.ones(len(stim_ind_match), dtype=int),
                ]
            )

        T = (
            stim_ons
            + n_items * (stim_dur + probe_dur)
            + delay
            + response_ons
            + response_dur
            + (n_items - 1) * (stim_offs + probe_offs)
        )

        # u is the input array
        u = np.zeros((self.n_channels, batch_size, T), dtype=np.float32)
        n_match = np.sum(labels)
        n_non_match = np.sum(labels == 0)
        u_match = np.zeros((self.n_channels, n_match, T), dtype=np.float32)
        u_non_match = np.zeros((self.n_channels, n_non_match, T), dtype=np.float32)

        # precalculate all stim_jits
        if stim_jit_dist == "uniform":
            isi_stim_match = (
                np.random.randint(
                    low=stim_jit[0], high=stim_jit[1], size=(n_match, (n_items - 1))
                )
                + stim_offs
            )
            isi_stim_non_match = (
                np.random.randint(
                    low=stim_jit[0], high=stim_jit[1], size=(n_non_match, (n_items - 1))
                )
                + stim_offs
            )

        elif stim_jit_dist == "poisson":
            isi_stim_match = np.int_(
                np.random.poisson(lam=stim_offs, size=(n_match, (n_items - 1)))
            )
            isi_stim_non_match = np.int_(
                np.random.poisson(lam=stim_offs, size=(n_non_match, (n_items - 1)))
            )

        else:
            print("WARNING, stim_jit_dist not supported!")
            print("use 'uniform' or 'poisson', continuing without stim jit")
            np.int_(isi_stim_match=np.ones((n_match, (n_items - 1) * 2)) * stim_offs)
            np.int_(
                isi_stim_non_match=np.ones((n_non_match, (n_items - 1) * 2)) * stim_offs
            )
        # precalculate all probe_jits
        if probe_jit_dist == "uniform":
            isi_probe_match = (
                np.random.randint(
                    low=probe_jit[0], high=probe_jit[1], size=(n_match, (n_items - 1))
                )
                + probe_offs
            )
            isi_probe_non_match = (
                np.random.randint(
                    low=probe_jit[0],
                    high=probe_jit[1],
                    size=(n_non_match, (n_items - 1)),
                )
                + probe_offs
            )
        elif probe_jit_dist == "poisson":
            isi_probe_match = np.int_(
                np.random.poisson(lam=probe_offs, size=(n_match, (n_items - 1)))
            )
            isi_probe_non_match = np.int_(
                np.random.poisson(lam=probe_offs, size=(n_non_match, (n_items - 1)))
            )
        else:
            print("WARNING, probe_jit_dist not supported!")
            print("use 'uniform' or 'poisson', continuing without probe jit")
            isi_probe_match = np.int_(np.ones((n_match, (n_items - 1))) * stim_offs)
            isi_probe_non_match = np.int_(
                np.ones((n_non_match, (n_items - 1))) * stim_offs
            )

        # randomize delay period to (0, delay) if random_delay
        if settings["random_delay"]:
            if settings["random_delay_per_tr"]:
                delays_match = np.random.choice(
                    np.arange(delay - settings["random_delay"], delay), n_match
                )
                delays_non_match = np.random.choice(
                    np.arange(delay - settings["random_delay"], delay), n_non_match
                )
            else:
                delay_len = np.random.choice(
                    np.arange(delay - settings["random_delay"], delay)
                )
                delays_match = np.int_(np.ones(n_match) * delay_len)
                delays_non_match = np.int_(np.ones(n_non_match) * delay_len)

        else:
            delays_match = np.int_(np.ones(n_match) * delay)
            delays_non_match = np.int_(np.ones(n_non_match) * delay)

        # TO DO! VECTORIZE! (Is that possible?)
        if settings["rand_stim_amp"]:
            low = 1 - settings["rand_stim_amp"]
            high = 1 + settings["rand_stim_amp"]

            # match trials:
            if n_match:
                if rand_ons:
                    onset_ind_match = np.random.choice(np.arange(-rand_ons, 0), n_match)
                else:
                    onset_ind_match = np.zeros(n_match)
                indices_stim = self.gen_match_trials(n_match, trial_ind_match)
                for ii in range(n_match):
                    probe_on = (
                        stim_ons
                        + stim_dur * n_items
                        + stim_offs * (n_items - 1)
                        + delays_match[ii]
                    )
                    for i in range(n_items):
                        stim_start = (
                            stim_ons + stim_dur * i + np.sum(isi_stim_match[ii, :i])
                        )
                        probe_start = (
                            probe_on + probe_dur * i + np.sum(isi_probe_match[ii, :i])
                        )
                        u_match[
                            indices_stim[ii, i], ii, stim_start : stim_start + stim_dur
                        ] = np.random.uniform(low, high)
                        u_match[
                            indices_stim[ii, i],
                            ii,
                            probe_start : probe_start + probe_dur,
                        ] = np.random.uniform(low, high)
                    u_match[:, ii] = np.roll(
                        u_match[:, ii], onset_ind_match[ii], axis=1
                    )

            # non match trials:
            if n_non_match:
                if rand_ons:
                    onset_ind_non_match = np.random.choice(
                        np.arange(-rand_ons, 0), n_non_match
                    )
                else:
                    onset_ind_non_match = np.zeros(n_non_match)
                indices_stim, indices_probes = self.gen_non_match_trials(
                    n_non_match, trial_ind_non_match
                )
                for ii in range(n_non_match):
                    probe_on = (
                        stim_ons
                        + stim_dur * n_items
                        + stim_offs * (n_items - 1)
                        + delays_non_match[ii]
                    )
                    for i in range(n_items):
                        stim_start = (
                            stim_ons + stim_dur * i + np.sum(isi_stim_non_match[ii, :i])
                        )
                        probe_start = (
                            probe_on
                            + probe_dur * i
                            + np.sum(isi_probe_non_match[ii, :i])
                        )
                        u_non_match[
                            indices_stim[ii, i], ii, stim_start : stim_start + stim_dur
                        ] = np.random.uniform(low, high)
                        u_non_match[
                            indices_probes[ii, i],
                            ii,
                            probe_start : probe_start + probe_dur,
                        ] = np.random.uniform(low, high)
                    u_non_match[:, ii] = np.roll(
                        u_non_match[:, ii], onset_ind_non_match[ii], axis=1
                    )
        else:
            # match trials:
            if n_match:
                if rand_ons:
                    onset_ind_match = np.random.choice(np.arange(-rand_ons, 0), n_match)
                else:
                    onset_ind_match = np.zeros(n_match)
                indices_stim = self.gen_match_trials(n_match, trial_ind_match)
                for ii in range(n_match):
                    probe_on = (
                        stim_ons
                        + stim_dur * n_items
                        + stim_offs * (n_items - 1)
                        + delays_match[ii]
                    )
                    for i in range(n_items):

                        stim_start = (
                            stim_ons + stim_dur * i + np.sum(isi_stim_match[ii, :i])
                        )
                        probe_start = (
                            probe_on + probe_dur * i + np.sum(isi_probe_match[ii, :i])
                        )

                        u_match[
                            indices_stim[ii, i], ii, stim_start : stim_start + stim_dur
                        ] = 1
                        u_match[
                            indices_stim[ii, i],
                            ii,
                            probe_start : probe_start + probe_dur,
                        ] = 1
                    u_match[:, ii] = np.roll(
                        u_match[:, ii], onset_ind_match[ii], axis=1
                    )
            # non match trials:
            if n_non_match:
                if rand_ons:
                    onset_ind_non_match = np.random.choice(
                        np.arange(-rand_ons, 0), n_non_match
                    )
                else:
                    onset_ind_non_match = np.zeros(n_non_match)
                indices_stim, indices_probes = self.gen_non_match_trials(
                    n_non_match, trial_ind_non_match
                )
                for ii in range(n_non_match):
                    probe_on = (
                        stim_ons
                        + stim_dur * n_items
                        + stim_offs * (n_items - 1)
                        + delays_non_match[ii]
                    )
                    for i in range(n_items):
                        stim_start = (
                            stim_ons + stim_dur * i + np.sum(isi_stim_non_match[ii, :i])
                        )
                        probe_start = (
                            probe_on
                            + probe_dur * i
                            + np.sum(isi_probe_non_match[ii, :i])
                        )
                        u_non_match[
                            indices_stim[ii, i], ii, stim_start : stim_start + stim_dur
                        ] = 1
                        u_non_match[
                            indices_probes[ii, i],
                            ii,
                            probe_start : probe_start + probe_dur,
                        ] = 1
                    u_non_match[:, ii] = np.roll(
                        u_non_match[:, ii], onset_ind_non_match[ii], axis=1
                    )

        u[:, labels == 1, :] = u_match
        u[:, labels == 0, :] = u_non_match
        delays_u = np.zeros(batch_size, dtype=int)
        stim_roll = np.zeros(batch_size, dtype=int)
        delays_u[labels == 1] = delays_match
        delays_u[labels == 0] = delays_non_match

        isi_stim = np.zeros((batch_size, (n_items - 1)))
        isi_stim[labels == 1] = isi_stim_match
        isi_stim[labels == 0] = isi_stim_non_match
        isi_probe = np.zeros((batch_size, (n_items - 1)))
        isi_probe[labels == 1] = isi_probe_match
        isi_probe[labels == 0] = isi_probe_non_match

        if n_match:
            stim_roll[labels == 1] = onset_ind_match
        if n_non_match:
            stim_roll[labels == 0] = onset_ind_non_match

    
        return u, labels, delays_u, stim_roll, isi_stim, isi_probe

    def generate_target(self, settings, label, T, delays, stim_roll, isi_probe=None):

        """
        Generate a set of targets matching inputs

        Args:
            settings: dictionary of task settings
            labels: labels for each trial (match or non match)
            T: trial duration in time steps
            delays: delays per trial
            stim_roll: randomised onset of first stim
            isi_probe: interstimulus intervals probe period
        Returns:
            z: target array
            m: mask (to emphasise specific times during training)
        """

        stim_offs = settings["stim_offs"]
        stim_dur = settings["stim_dur"]
        probe_offs = settings["probe_offs"]
        probe_dur = settings["probe_dur"]
        batch_size = len(label)
        n_items = settings["n_items"]
        response_ons = settings["response_ons"]
        response_dur = settings["response_dur"]
        probe_gain = settings["probe_gain"]

        if stim_roll is None:
            stim_roll = 0
        stim_ons = settings["stim_ons"] + stim_roll
        if isi_probe is None:
            isi_probe = np.ones((batch_size, (n_items - 1))) * probe_offs

        response_times = np.int_(
            stim_ons
            + n_items * (stim_dur + probe_dur)
            + stim_offs * (n_items - 1)
            + np.sum(isi_probe, axis=1)
            + delays
            + response_ons
        )
        # output array
        z = np.zeros((T, self.out_channels, batch_size), dtype=np.float32)


        if self.out_channels == 3:
            mask = np.ones((T, self.out_channels, batch_size), dtype=np.float32)

            for i in range(batch_size):
                response_time = response_times[i]
                z[:response_time, 0, i] = 1
                z[response_time + response_dur :, 0, i] = 1

                if label[i] == 1:
                    z[response_time : response_time + response_dur, 1, i] = 1
                else:
                    z[response_time : response_time + response_dur, 2, i] = 1
          
        elif self.out_channels == 1:
            mask = np.zeros((T, self.out_channels, batch_size), dtype=np.float32)

            for i in range(batch_size):
                response_time = response_times[i]
                if label[i] == 1:
                    z[response_time : response_time + response_dur, 0, i] = 1
                else:
                    z[response_time : response_time + response_dur, 0, i] = -1

                mask[response_time : response_time + response_dur, 0, i] = probe_gain

          
        return z, mask
