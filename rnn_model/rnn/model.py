import os, sys
import sys
sys.path.insert(0,'..')
sys.path.append(os.getcwd())
import numpy as np
import tensorflow.compat.v1 as tf
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append(file_dir+"/..")

from analysis.tf_utils import scalogram
import scipy.io
import datetime
import time
import wandb


"""
Recurrent neural network class
"""
class RNN:
    def __init__(self):
       print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
       tf.disable_eager_execution()

    def initialize_model(self, model_params): 
        """
        Initialize model
        Args:
            model_params: dictionary of model parameters
        """

        self.N = model_params["N"]
        self.n_channels = model_params["n_channels"]
        self.apply_dale = model_params["apply_dale"]
        self.out_channels = model_params["out_channels"]

        # create a dictionary to store initial states
        self.initializer = {}

        # Input task weight matrix
        w_in = np.zeros((self.N, self.n_channels), dtype=np.float32)
        idx = np.array(
            np.where(np.random.rand(self.N, self.n_channels) < model_params["P_in"])
        )
        w_in[idx[0], idx[1]] = np.random.randn(len(idx[0])) * np.sqrt(
            1 / self.n_channels * model_params["P_in"]
        )
        self.initializer["w_in"] = w_in

        # Recurrent  Weight matrix, masks and bias
        w_rec = np.zeros((self.N, self.N), dtype=np.float32)
        dale_mask = np.eye(self.N, dtype=np.float32)
        conn_mask = np.ones((self.N, self.N), dtype=np.float32)

        # if no_autapses is true the diagonal will be all zeros
        if model_params["no_autapses"]:
            rec_idx = np.where(
                np.random.rand(self.N, self.N)
                < (model_params["P_rec"] * ((self.N + 1) / self.N))
            )
            np.fill_diagonal(conn_mask, 0)
        else:
            rec_idx = np.where(np.random.rand(self.N, self.N) < model_params["P_rec"])

        # initialize with weights drawn from either Gaussian or Gamma distribution
        if model_params["w_dist"] == "Gauss":
            w_rec[rec_idx[0], rec_idx[1]] = (
                np.random.normal(0, 1, len(rec_idx[0]))
                * model_params["spec_rad"]
                / np.sqrt(model_params["P_rec"] * self.N)
            )
        elif model_params["w_dist"] == "Gamma":
            w_rec[rec_idx[0], rec_idx[1]] = np.random.gamma(2, 0.5, len(rec_idx[0]))
            if model_params["spectr_norm"] == False:
                print(
                    "WARNING: analytic normalisation not implemented, setting spectral normalisation to TRUE"
                )
                model_params["spectr_norm"] = True
            if self.apply_dale == False:
                print(
                    "WARNING: Gamma distribution is all positive, use only with Dale's law, setting Dale's law to TRUE"
                )
                self.apply_dale == True

        else:
            print("WARNING: initialization not implemented, use Gauss or Gamma")
            print("continuing with Gauss")
            w_rec[rec_idx[0], rec_idx[1]] = (
                np.random.normal(0, 1, len(rec_idx[0]))
                * model_params["spec_rad"]
                / np.sqrt(model_params["P_rec"] * self.N)
            )

        # only excitory or inhibitory outgoing connections per neuron
        if self.apply_dale:
            dale_mask[-int(self.N * model_params["P_inh"]) :] *= -1
            w_rec = np.abs(w_rec)

            # Balanced DL (expectation input = 0)
            if model_params["balance_DL"]:
                n_inh = int(self.N * model_params["P_inh"])
                EIratio = (1 - model_params["P_inh"]) / (model_params["P_inh"])
                w_rec[:, -n_inh:] *= EIratio
                if model_params["rm_outlier_DL"]:
                    ex_u = np.sum(w_rec[:, :-n_inh], axis=1)
                    in_u = np.sum(w_rec[:, -n_inh:], axis=1)
                    ratio = ex_u / in_u
                    w_rec[:, :-n_inh] /= np.expand_dims(ratio, 1)
                b = np.sqrt((1 / (1 - (2 * model_params["P_rec"]) / np.pi)) / EIratio)
                w_rec *= b

        # set to desired spectral radius
        if model_params["spectr_norm"]:
            w_rec = (
                model_params["spec_rad"]
                * w_rec
                / np.max(np.abs((np.linalg.eigvals(dale_mask.dot(w_rec) * conn_mask))))
            )
        print(
            "spectral_rad: "
            + str(np.max(abs(np.linalg.eigvals(dale_mask.dot(w_rec) * conn_mask))))
        )
        self.initializer["w_in_scale"] = 1.0
        self.initializer["w_out_scale"] = 1.0

        # add recurrent weights to dictionary
        self.initializer["w_rec"] = np.float32(w_rec)
        self.initializer["dale_mask"] = dale_mask
        self.initializer["conn_mask"] = conn_mask
        self.initializer["b_rec"] = np.zeros((self.N, 1), dtype=np.float32)

        # add output weight matrix and bias
        if model_params["1overN_out"]:
            self.initializer["w_out"] = np.float32(
                np.random.randn(self.out_channels, self.N) / self.N
            ) 
        else:
            self.initializer["w_out"] = np.float32(
                np.random.randn(self.out_channels, self.N) * np.sqrt(1 / self.N)
            )
        self.initializer["b_out"] = np.zeros((self.out_channels, 1), dtype=np.float32)

        # add distribution of taus
        self.initializer["taus_gaus"] = np.float32(np.random.randn(self.N, 1))
        self.initializer["init_state"] = np.float32(
            0 + 0.01 * np.random.randn(self.N, 1)
        )

    def rnn_cell(self, state, rnn_in):
    
        """
        Recurrent neural network cell, computes one time step forward 
        Args:
            state: hidden state
            rnn_in: external input
        Returns:
            next_state: hidden state at next time step
        """


        # calculate effective rnn weights
        if self.apply_dale == True:
            w_rec_eff = tf.nn.relu(self.w_rec)
            w_rec_eff = tf.matmul(w_rec_eff, self.dale_mask)

        else:
            w_rec_eff = self.w_rec
        # dales law
        # further constraints on connectivity
        w_rec_eff = tf.multiply(w_rec_eff, self.conn_mask)
        w_in_eff = self.w_in*self.w_in_scale
        # Update neural activity and short-term synaptic plasticity values
        r_post = self.transfer_function(state)
     
        # calculate next step
        new_state = (
            tf.multiply((1 - self.alpha), state)
            + tf.multiply(
                self.alpha,
                (
                    (tf.matmul(w_rec_eff, r_post) + self.b_rec)
                    + tf.matmul(w_in_eff, rnn_in)
                ),
            )
            + tf.sqrt(2.0 * self.alpha)
            * self.rec_noise
            * tf.random.normal(state.shape, dtype=tf.float32, mean=0.0, stddev=1.0)
        )

        return new_state

    def output_timestep(self, state):
    
        """
        Computes output, linear projection of network rates
        Args:
            state: hidden state
        Returns:
            output: linear projection of network rates
        """

        output = tf.matmul(self.w_out_eff, self.transfer_function(state)) + self.b_out

        return output

    def forward_pass(self):

        """
        Computes forward pass through all timesteps

        Returns:
            x: model hidden states
            o: model output
        """

        x = []  # currents
        o = []  # output (i.e. weighted linear sum of rates, r)

        if self.rand_init_x:
            state = tf.random.normal(
                [self.N, self.batch_size], dtype=tf.float32, mean=0.0, stddev=0.1
            )
        else:
            state = tf.tile(self.init_state, [1, self.batch_size])
        rnn_inputs = tf.unstack(self.stim, axis=2)
        self.w_out_eff = self.w_out
       
        # loop through all inputs
        for rnn_input in rnn_inputs:
            state = self.rnn_cell(state, rnn_input)
            out = self.output_timestep(state)
            x.append(state)
            o.append(out)

        return x, tf.stack(o)


    def construct_graph(self, training_params):

        """
        Construct the tensorflow graph
        
        Args: 
            training_params: dictionary of training parameters
        """

        print("Start building TF graph")

        # Input node
        self.stim = tf.compat.v1.placeholder(
            tf.float32,
            [self.n_channels, self.batch_size, self.N_steps],
            name="u",
        )

        # Target node
        self.z = tf.compat.v1.placeholder(
            tf.float32,
            [self.N_steps, self.out_channels, self.batch_size],
            name="target",
        )

        # Loss mask (to put emphasis on certain time periods, e.g. decision period)
        self.loss_mask = tf.compat.v1.placeholder(
            tf.float32,
            [self.N_steps, self.out_channels, self.batch_size],
            name="loss_mask",
        )

        # Initialize the decay synaptic time-constants (gaussian random).
        # This vector will go through the sigmoid transfer function to bound it between tau lims
        tau_lims = training_params["tau_lims"]
        if len(tau_lims) > 1:
            self.taus_gaus = tf.get_variable(
                "taus_gaus",
                initializer=self.initializer["taus_gaus"],
                dtype=tf.float32,
                trainable=training_params["train_taus"],
            )
            taus_sig = (
                tf.sigmoid(self.taus_gaus) * (tau_lims[1] - tau_lims[0]) + tau_lims[0]
            )

        elif len(tau_lims) == 1:
            self.taus_gaus = tf.get_variable(
                "taus_gaus",
                initializer=self.initializer["taus_gaus"],
                dtype=tf.float32,
                trainable=False,
            )
            taus_sig = tau_lims[0]

        self.alpha = training_params["deltaT"] / taus_sig

        # Initialize recurrent weight matrix, mask, input & output weight matrices
        self.w_rec = tf.get_variable(
            "w_rec",
            initializer=self.initializer["w_rec"],
            dtype=tf.float32,
            trainable=training_params["train_w_rec"],
        )
        self.dale_mask = tf.constant(
            self.initializer["dale_mask"]
        )  # , dtype=tf.float32, trainable=False)
     
        self.w_in = tf.get_variable(
            "w_in",
            initializer=self.initializer["w_in"],
            dtype=tf.float32,
            trainable=training_params["train_w_in"],
        )
        self.w_in_scale = tf.get_variable(
            "w_in_scale",
            initializer=self.initializer["w_in_scale"],
            dtype=tf.float32,
            trainable=training_params["train_w_in_scale"],
        )
        self.w_out = tf.get_variable(
            "w_out",
            initializer=self.initializer["w_out"],
            dtype=tf.float32,
            trainable=training_params["train_w_out"],
        )
        self.conn_mask = tf.constant(self.initializer["conn_mask"])
        self.b_out = tf.get_variable(
            "b_out",
            initializer=self.initializer["b_out"],
            dtype=tf.float32,
            trainable=training_params["train_b_out"],
        )
        self.b_rec = tf.get_variable(
            "b_rec",
            initializer=self.initializer["b_rec"],
            dtype=tf.float32,
            trainable=training_params["train_b_rec"],
        )
        self.init_state = tf.compat.v1.get_variable(
            "init_state",
            initializer=self.initializer["init_state"],
            dtype=tf.float32,
            trainable=training_params["train_init_state"],
        )
        print("initialised variables ")

        # forward pass through all timesteps

        self.x, self.o = self.forward_pass()
        print("initialised forward pass")
        self.loss_op(training_params)
        print("initialised loss")

    def loss_op(self, training_params):

        """
        Execute the loss operation
        
        Args: 
            training_params: dictionary of training parameters
    
        """

        #Compute the used weight matrices (with masks etc)

        ww = tf.nn.relu(self.w_rec)
        ww = tf.matmul(ww, self.dale_mask)
        ww = tf.multiply(ww, self.conn_mask)
        ww_in = self.w_in * self.w_in_scale
        
        # Loss function
        self.loss = tf.zeros(1)
        if training_params["loss"] == "sce":
            self.perf_loss = tf.math.reduce_sum(
                tf.reduce_mean(self.loss_mask, axis=1)
                * tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.o, labels=self.z, axis=1
                )
            ) / (tf.math.reduce_sum(self.loss_mask))
        elif self.out_channels == 2:
            self.perf_loss = tf.math.reduce_sum(
                (
                    tf.math.reduce_sum(
                        self.loss_mask * tf.square(self.o - self.z), axis=(0, 2)
                    )
                )
                / (tf.math.reduce_sum(self.loss_mask, axis=(0, 2)))
            )

        elif training_params["loss"] == "l2":
            self.perf_loss = tf.math.reduce_sum(
                self.loss_mask * tf.square(self.o - self.z)
            ) / (tf.math.reduce_sum(self.loss_mask))


        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        self.spike_loss = tf.reduce_mean(tf.square(self.transfer_function(self.x)))

        # L2 penalty term on weights
        self.out_weight_loss = tf.reduce_mean(tf.square(self.w_out))
        self.rec_weight_loss = tf.reduce_mean(tf.square(ww))
        self.in_weight_loss = tf.reduce_mean(tf.square(ww_in))

        # Loss term on non oscillating firing rates
        LFP = []
        t1 = 0
        t2 = self.N_steps  # -130
    
        inh_mask = np.eye(self.N, dtype=np.float32)
        if training_params["osc_reg_inh"]:
            inh_mask[:training_params["osc_reg_inh"]] = 0.0
        ww_m = tf.matmul(ww, tf.convert_to_tensor(inh_mask))

        for x in self.x:
            LFP.append(tf.matmul(tf.abs(ww_m), self.transfer_function(x)))
        LFPt = tf.stack(LFP[t1:t2])
        LFPt = tf.reduce_mean(LFPt, axis=1) #take mean input current over neurons
        x_u, x_var = tf.nn.moments(LFPt, [0])  # mean and var over time axis
        x_z = (LFPt - x_u) / (tf.math.sqrt(2 * x_var + 1e-10))[t1:t2]
        tstep = training_params["deltaT"] / 1000

        
        trtime = np.arange(0, self.N_steps * tstep, tstep, dtype=np.float32)[
            t1:t2
        ]
        trtime = tf.constant(trtime)

        sin = tf.math.sin(training_params["lossF"] * 2 * np.pi * trtime)
        cos = tf.math.cos(training_params["lossF"] * 2 * np.pi * trtime)

        cosFT = (tf.tensordot(cos, x_z, axes=[[0], [0]]) / (t2 - t1)) ** 2
        sinFT = (tf.tensordot(sin, x_z, axes=[[0], [0]]) / (t2 - t1)) ** 2

        osc_loss = tf.math.sqrt(sinFT + cosFT)


        self.osc_loss = -tf.reduce_mean(osc_loss)
        # Loss term on non oscillating firing rates

        # total loss
        self.loss = (
            self.perf_loss
            + training_params["spike_cost"] * self.spike_loss
            + training_params["out_weight_cost"] * self.out_weight_loss
            + training_params["rec_weight_cost"] * self.rec_weight_loss
            + training_params["in_weight_cost"] * self.in_weight_loss
            + training_params["osc_cost"] * self.osc_loss
        )

        # with tf.name_scope('ADAM'):
        opt = tf.train.AdamOptimizer(learning_rate=training_params["learning_rate"])

        grads_and_vars = opt.compute_gradients(self.loss)

        # Apply any applicable weights masks to the gradient and clip
        capped_gvs = []
        for grad, var in grads_and_vars:
            if "w_rec" in var.op.name:
               capped_gvs.append(
                (tf.clip_by_norm(grad, training_params["clip_max_grad_val"]), var)
            )

        self.training_op = opt.apply_gradients(capped_gvs)

    def train(
        self,
        training_params,
        model_params,
        trial_gen,
        gpu,
        gpu_frac,
        out_dir,
        name=None,
        new_run=True,
        sync_wandb=True,
        clear_tf=True,
    ):

        """
        Run the training loop
        
        Args:
            training_params: dictionary of training parameters
            model_params: dictionary of model parameters
            trial_gen: instance of trial generator object
            gpu: which gpu to use
            gpu_frac: scalar in [0,1], how much of the simmulation to use on gpu.
            out_dir: string, where to save models
            name: string, model name for saving
            new_run: Bool, whether starting a new model or loading a model
            sync_wandb, Bool, whether or not to synchronise with WandB
            clear_tf, clear tensorflow graph at the end

        """
        self.sync_wandb = sync_wandb
        if sync_wandb:
            if new_run:
                wandb.init(
                    project="phase-coding",
                    group="osc_driven_sweep",
                    config={**model_params, **training_params},
                )  # , reinit=True)
            config = wandb.config

            """overwrite config with sweep params"""
            """if not running a sweep this changes nothing"""
            print("OSC COST" + str(config.osc_cost))
            print("LOSSF" + str(config.lossF))
            training_params["osc_cost"] = config.osc_cost
            if name is not None:
                name = name + str(config.lossF) + "cost" + str(config.osc_cost)
            training_params["lossF"] = config.lossF
            training_params["tau_lims"] = config.tau_lims
            training_params["learning_rate"] = config.learning_rate     
            training_params["train_w_in_scale"] = config.train_w_in_scale
            training_params["stim_offs"] = config.stim_offs
            training_params["probe_offs"] = config.stim_offs
        self.activation = training_params["activation"]

        # Transfer function options
        if self.activation == "sigmoid":
            self.transfer_function = tf.sigmoid
        elif self.activation == "relu":
            print("activation set to ReLU")
            self.transfer_function = tf.nn.relu
            # use he initialization for ReLU
            if new_run:
                self.initializer["w_in"] = self.initializer["w_in"] * np.sqrt(2)
                self.initializer["w_out"] = self.initializer["w_out"] * np.sqrt(2)
        elif self.activation == "softplus":
            self.transfer_function = tf.nn.softplus
        elif self.activation == "tanh":
            self.transfer_function = tf.tanh

        self.batch_size = training_params["batch_size"]
        self.rec_noise = training_params["rec_noise"]
        self.rand_init_x = training_params["rand_init_x"]

        # store validation trial_ind
        self.val_ind = trial_gen.val_ind
        self.train_ind = trial_gen.train_ind

        # keep track of losses and performance during training
        start_time = time.time()
        self.train_loss_list = []
        self.train_perf_list = []
        self.val_loss_list = []
        self.val_perf_list = []

        # make dictionary of initial states
        self.pre_train_state = self.initializer.copy()

        # loop through all delays (curriculum learning)
        for delay in training_params["delays"]:
            print(delay)
            print(training_params["delays"][0])
            # if we do curriculum learning, reconstruct the graph with a new delay sequentially
            if int(delay) is not int(training_params["delays"][0]):
                print("reseting TF default graph")
                tf.reset_default_graph()
                self.initializer["w_in"] = t_w_in
                self.initializer["w_in_scale"] = t_w_in_scale
                self.initializer["w_rec"] = t_w_rec
                self.initializer["dale_mask"] = t_dale_mask
                self.initializer["conn_mask"] = t_conn_mask
                self.initializer["b_rec"] = t_b_rec
                self.initializer["b_out"] = t_b_out
                self.initializer["w_out"] = t_w_out
                self.initializer["taus_gaus"] = t_taus_gaus
                self.initializer["init_state"] = t_init_state

            self.response_start = (
                training_params["stim_ons"]
                + training_params["n_items"]
                * (training_params["stim_dur"] + training_params["probe_dur"])
                + delay
                + training_params["response_ons"]
                + (training_params["n_items"] - 1)
                * (training_params["stim_offs"] + training_params["probe_offs"])
            )
            self.N_steps = (
                self.response_start
                + training_params["response_dur"]
                + training_params["response_ons"]
            )

            # construct tensorflow graph and initialize
            self.construct_graph(training_params)
            print("Constructed the TF graph...")
            init = tf.global_variables_initializer()
            print("initialized variables")
            self.sess = tf.Session(
                config=tf.ConfigProto(
                    gpu_options=set_gpu(gpu, gpu_frac),
                    intra_op_parallelism_threads=10,
                    inter_op_parallelism_threads=10,
                    allow_soft_placement=True,
                )
            )  # True))
            with self.sess.as_default():
                print("Training started...")
                init.run()
                for tr in range(training_params["n_trials"]):

                    # Generate input
                    (
                        u,
                        label,
                        trial_delays,
                        stim_roll,
                        _,
                        isi_probe,
                    ) = trial_gen.generate_input(training_params, delay, val=False)
                    target, mask = trial_gen.generate_target(
                        training_params,
                        label,
                        self.N_steps,
                        trial_delays,
                        stim_roll,
                        isi_probe,
                    )
                    self.sess.run(
                        [self.training_op],
                        feed_dict={
                            self.stim: u.astype(np.float32),
                            self.z: target.astype(np.float32),
                            self.loss_mask: mask,
                        },
                    )
                    if (tr - 1) % training_params["eval_freq"] == 0:
                        # Run Validation Set
                        if len(trial_gen.val_ind) > 0:
                            eval_perf = np.zeros((1, training_params["eval_tr"]))
                            eval_losses = np.zeros((1, training_params["eval_tr"]))
                            for ii in range(eval_perf.shape[-1]):

                                # generate task
                                (
                                    eval_u,
                                    eval_label,
                                    eval_delays,
                                    stim_roll,
                                    _,
                                    isi_probe,
                                ) = trial_gen.generate_input(
                                    training_params, delay, val=True
                                )
                                eval_target, mask = trial_gen.generate_target(
                                    training_params,
                                    eval_label,
                                    self.N_steps,
                                    eval_delays,
                                    stim_roll,
                                    isi_probe,
                                )

                                # run trial
                                eval_o, eval_l, t_w_rec = self.sess.run(
                                    [self.o, self.loss, self.w_rec],
                                    feed_dict={
                                        self.stim: eval_u,
                                        self.z: eval_target,
                                        self.loss_mask: mask,
                                    },
                                )
                                # check for NAN in w matrix, stop training if so
                                if np.isnan(t_w_rec[1]).any():
                                    print("NAN FOUND IN W MATRIX")
                                    exit()

                                # store average loss over batches
                                eval_losses[0, ii] = eval_l

                                # calculate average performance over batches
                                correct = 0
                                response_time = np.int_(
                                    training_params["stim_ons"]
                                    + stim_roll
                                    + training_params["n_items"]
                                    * (
                                        training_params["stim_dur"]
                                        + training_params["probe_dur"]
                                    )
                                    + eval_delays
                                    + 1
                                    + training_params["response_ons"]
                                    + (training_params["n_items"] - 1)
                                    * (training_params["stim_offs"])
                                    + np.sum(isi_probe, axis=1)
                                    + training_params["response_ons"]
                                )

                                if self.out_channels == 3:
                                    for i in range(self.batch_size):
                                        if eval_label[i] == 1:
                                            correct += np.max(
                                                eval_o[
                                                    response_time[i] : response_time[i]
                                                    + training_params["response_dur"],
                                                    1,
                                                    i,
                                                ],
                                                axis=0,
                                            ) > np.max(
                                                eval_o[
                                                    response_time[i] : response_time[i]
                                                    + training_params["response_dur"],
                                                    ::2,
                                                    i,
                                                ],
                                                axis=(0, 1),
                                            )
                                        else:
                                            correct += np.max(
                                                eval_o[
                                                    response_time[i] : response_time[i]
                                                    + training_params["response_dur"],
                                                    2,
                                                    i,
                                                ],
                                                axis=0,
                                            ) > np.max(
                                                eval_o[
                                                    response_time[i] : response_time[i]
                                                    + training_params["response_dur"],
                                                    :2,
                                                    i,
                                                ],
                                                axis=(0, 1),
                                            )

                                else:
                                    for i in range(self.batch_size):
                                        sign = np.sign(
                                            eval_o[
                                                response_time[i] : response_time[i]
                                                + training_params["response_dur"],
                                                0,
                                                i,
                                            ]
                                        )
                                        if eval_label[i] == 1:
                                            if np.sum(sign == 1) > np.sum(sign == -1):
                                                correct += 1
                                        else:
                                            if np.sum(sign == -1) > np.sum(sign == 1):
                                                correct += 1

                                eval_perf[0, ii] = correct / self.batch_size

                            eval_perf_mean = np.nanmean(eval_perf, 1)
                            eval_loss_mean = np.nanmean(eval_losses, 1)
                            self.val_perf_list.append(eval_perf_mean)
                            self.val_loss_list.append(eval_loss_mean)
                            print(
                                "Trial: "
                                + str(tr)
                                + ", VAL Perf: %.2f, Loss: %.5f"
                                % (eval_perf_mean, eval_loss_mean)
                            )
                            if sync_wandb:
                                wandb.log(
                                    {
                                        "EVALacc": eval_perf_mean,
                                        "EVALloss": eval_loss_mean,
                                    }
                                )

                        # Run Training Set
                        eval_perf = np.zeros((1, training_params["eval_tr"]))
                        eval_losses = np.zeros((1, training_params["eval_tr"]))
                        for ii in range(eval_perf.shape[-1]):

                            # generate task
                            (
                                eval_u,
                                eval_label,
                                eval_delays,
                                stim_roll,
                                _,
                                isi_probe,
                            ) = trial_gen.generate_input(
                                training_params, delay, val=False
                            )
                            eval_target, mask = trial_gen.generate_target(
                                training_params,
                                eval_label,
                                self.N_steps,
                                eval_delays,
                                stim_roll,
                                isi_probe,
                            )

                            # run trial
                            eval_o, eval_l, t_w_rec = self.sess.run(
                                [self.o, self.loss, self.w_rec],
                                feed_dict={
                                    self.stim: eval_u,
                                    self.z: eval_target,
                                    self.loss_mask: mask,
                                },
                            )
                            # check for NAN in w matrix, stop training if so
                            if np.isnan(t_w_rec[1]).any():
                                print("NAN FOUND IN W MATRIX")
                                exit()

                            # store average loss over batches
                            eval_losses[0, ii] = eval_l

                            # calculate average performance over batches
                            correct = 0
                            response_time = np.int_(
                                training_params["stim_ons"]
                                + stim_roll
                                + training_params["n_items"]
                                * (
                                    training_params["stim_dur"]
                                    + training_params["probe_dur"]
                                )
                                + eval_delays
                                + 1
                                + training_params["response_ons"]
                                + (training_params["n_items"] - 1)
                                * (training_params["stim_offs"])
                                + np.sum(isi_probe, axis=1)
                                + training_params["response_ons"]
                            )

                            if self.out_channels == 3:
                                for i in range(self.batch_size):
                                    if eval_label[i] == 1:
                                        correct += np.max(
                                            eval_o[
                                                response_time[i] : response_time[i]
                                                + training_params["response_dur"],
                                                1,
                                                i,
                                            ],
                                            axis=0,
                                        ) > np.max(
                                            eval_o[
                                                response_time[i] : response_time[i]
                                                + training_params["response_dur"],
                                                ::2,
                                                i,
                                            ],
                                            axis=(0, 1),
                                        )
                                    else:
                                        correct += np.max(
                                            eval_o[
                                                response_time[i] : response_time[i]
                                                + training_params["response_dur"],
                                                2,
                                                i,
                                            ],
                                            axis=0,
                                        ) > np.max(
                                            eval_o[
                                                response_time[i] : response_time[i]
                                                + training_params["response_dur"],
                                                :2,
                                                i,
                                            ],
                                            axis=(0, 1),
                                        )

                            else:
                                for i in range(self.batch_size):
                                    sign = np.sign(
                                        eval_o[
                                            response_time[i] : response_time[i]
                                            + training_params["response_dur"],
                                            0,
                                            i,
                                        ]
                                    )
                                    if eval_label[i] == 1:
                                        if np.sum(sign == 1) > np.sum(sign == -1):
                                            correct += 1
                                    else:
                                        if np.sum(sign == -1) > np.sum(sign == 1):
                                            correct += 1

                            eval_perf[0, ii] = correct / self.batch_size

                        eval_perf_mean = np.nanmean(eval_perf, 1)
                        eval_loss_mean = np.nanmean(eval_losses, 1)
                        self.train_perf_list.append(eval_perf_mean)
                        self.train_loss_list.append(eval_loss_mean)
                        print(
                            "Trial: "
                            + str(tr)
                            + ", TRAIN Perf: %.2f, Loss: %.5f"
                            % (eval_perf_mean, eval_loss_mean)
                        )
                        if sync_wandb:
                            wandb.log({"acc": eval_perf_mean, "loss": eval_loss_mean})

                        # finished training, stop and save
                        if (
                            eval_loss_mean <= training_params["loss_threshold"]
                            and eval_perf_mean >= training_params["acc_threshold"]
                        ):
                            (   x,
                                t_init_state,
                                t_w_in,
                                t_w_in_scale,
                                t_w_rec,
                                t_w_out,
                                t_b_rec,
                                t_b_out,
                                t_taus_gaus,
                                t_dale_mask,
                                t_conn_mask,
                      
                            ) = self.sess.run(
                                [   self.x,
                                    self.init_state,
                                    self.w_in,
                                    self.w_in_scale,
                                    self.w_rec,
                                    self.w_out,
                                    self.b_rec,
                                    self.b_out,
                                    self.taus_gaus,
                                    self.dale_mask,
                                    self.conn_mask],
                                    feed_dict={
                                    self.stim: eval_u,
                                    self.z: eval_target,
                                    self.loss_mask: mask,
                                     }
                                
                            )  # , feed_dict = \
                            # {self.stim: eval_u, self.z: eval_target})

                            freq = lfp_p(x, t_conn_mask, t_dale_mask, t_w_rec, training_params['deltaT'], self.N_steps, training_params['activation'])
                            print("freq: " + str(freq))
                            if sync_wandb:
                                wandb.log({"freq": freq})
                                

                            self.save_model(
                                out_dir,
                                training_params,
                                t_init_state,
                                t_w_in,
                                t_w_in_scale,
                                t_w_rec,
                                t_w_out,
                                t_b_rec,
                                t_b_out,
                                t_taus_gaus,
                                t_dale_mask,
                                t_conn_mask,
                                eval_perf_mean,
                                tr,
                                delay,
                                name,
                            )
                            print("saved final model")

                            break

                    # save model in every saving_freq trials
                    if (
                        (tr - 1) % training_params["saving_freq"] == 0
                        and (tr - 1 != 0)
                        or tr == (training_params["n_trials"] - 1)
                    ):
                        (   x,
                            t_init_state,
                            t_w_in,
                            t_w_in_scale,
                            t_w_rec,
                            t_w_out,
                            t_b_rec,
                            t_b_out,
                            t_taus_gaus,
                            t_dale_mask,
                            t_conn_mask,
                    
                        ) = self.sess.run(
                            [   self.x,
                                self.init_state,
                                self.w_in,
                                self.w_in_scale,
                                self.w_rec,
                                self.w_out,
                                self.b_rec,
                                self.b_out,
                                self.taus_gaus,
                                self.dale_mask,
                                self.conn_mask],
                                feed_dict={
                                self.stim: eval_u,
                                self.z: eval_target,
                                self.loss_mask: mask,
                                    }
                            
                        )  # , feed_dict = \
                        # {self.stim: eval_u, self.z: eval_target})

                        freq = lfp_p(x, t_conn_mask, t_dale_mask, t_w_rec, training_params['deltaT'], self.N_steps, training_params['activation'])
                        print("freq: " + str(freq))
                        if sync_wandb:
                            wandb.log({"freq": freq})
                            

                        self.save_model(
                            out_dir,
                            training_params,
                            t_init_state,
                            t_w_in,
                            t_w_in_scale,
                            t_w_rec,
                            t_w_out,
                            t_b_rec,
                            t_b_out,
                            t_taus_gaus,
                            t_dale_mask,
                            t_conn_mask,
                            eval_perf_mean,
                            tr,
                            delay,
                            name,
                        )
                        print("saving model")

            elapsed_time = time.time() - start_time
            print("time: " + str(elapsed_time))
        if clear_tf:
            self.sess.close()
            tf.reset_default_graph()

        if sync_wandb:
            wandb.finish()

    def load_model(self, model_dir):

        """
        Method to load trained network

        Args: 
            model_dir: dictionary containing model parameters
        """

        settings = scipy.io.loadmat(model_dir)
        self.N = settings["N"][0][0]
        self.n_channels = settings["n_channels"][0][0]
        self.out_channels = settings["out_channels"][0][0]
        self.apply_dale = settings["apply_dale"]
        self.activation = settings["activation"]
        self.initializer = {}
        self.initializer["w_in"] = settings["t_w_in"]
        self.initializer["w_rec"] = settings["t_w_rec"]
        self.initializer["dale_mask"] = settings["t_dale_mask"]
        self.initializer["conn_mask"] = settings["t_conn_mask"]
        self.initializer["b_rec"] = settings["t_b_rec"]
        self.initializer["b_out"] = settings["t_b_out"]
        self.initializer["w_out"] = settings["t_w_out"]
        self.initializer["taus_gaus"] = settings["t_taus_gaus"]
        self.initializer["init_state"] = settings["t_init_state"]
        self.rec_noise = settings["rec_noise"]

        if "t_w_in_scale" in settings.keys():
            self.initializer["w_in_scale"] = settings["t_w_in_scale"]
        else:
            self.initializer["w_in_scale"] = 1.0


    def save_model(
        self,
        out_dir,
        training_params,
        t_init_state,
        t_w_in,
        t_w_in_scale,
        t_w_rec,
        t_w_out,
        t_b_rec,
        t_b_out,
        t_taus_gaus,
        t_dale_mask,
        t_conn_mask,
        eval_perf_mean,
        tr,
        delay,
        name=None,
    ):

        """
        Method to store trained network
        Args:
            out_dir: String, where to store model
            training_params: dictionary of parameters used for training
            t_...: parameters of trained network
            eval_perf_mean: mean eval performance per epoch
            tr: amount of trials before convergence
            delay: delay used in last trials
            name: String, model name
        """

        var = {}

        # general values
        var["N"] = self.N
        var["n_channels"] = self.n_channels
        var["out_channels"] = self.out_channels
        var["apply_dale"] = self.apply_dale
        var["val_ind"] = self.val_ind
        var["train_ind"] = self.train_ind

        # after training
        var["t_init_state"] = t_init_state
        var["t_w_in"] = t_w_in
        var["t_w_in_scale"] = t_w_in_scale
        var["t_w_rec"] = t_w_rec
        var["t_w_out"] = t_w_out
        var["t_b_rec"] = t_b_rec
        var["t_b_out"] = t_b_out
        var["t_taus_gaus"] = t_taus_gaus
        var["t_dale_mask"] = t_dale_mask
        var["t_conn_mask"] = t_conn_mask

        # training_summary
        var["train_perf_list"] = self.train_perf_list
        var["train_loss_list"] = self.train_loss_list
        var["val_perf_list"] = self.val_perf_list
        var["val_loss_list"] = self.val_loss_list
        var["tr"] = tr

        # save initial state
        var["pre_training_state"] = self.pre_train_state

        # add initialization and trianing parameters to file
        combined = {**self.initializer, **training_params, **var}

        # create unique filename
        fname_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
        if name is None:
            fname = "N_items_{}_N_{}_Delay_{}_Acc_{}_tr{}_{}.mat".format(
                training_params["n_items"],
                self.N,
                delay,
                eval_perf_mean[0],
                tr,
                fname_time,
            )
        else:
            fname = name

        # export to .mat file
        scipy.io.savemat(os.path.join(out_dir, fname), combined)
        if self.sync_wandb:
            wandb.save(os.path.join(out_dir, fname))

    def predict(self, settings, u):

        """
        Numpy implementation for easier analysis of trained models

        Args: 
            settings, dictionary of model parameters
        
        """

        # load all settings
        T = u.shape[2]
        deltaT = settings["deltaT"]
        batch_size = settings["batch_size"]
        tau_lims = settings["tau_lims"]


        w_in = self.initializer["w_in"]
        w_in_scale = self.initializer["w_in_scale"]
        w_rec = self.initializer["w_rec"]
        dale_mask = self.initializer["dale_mask"]
        conn_mask = self.initializer["conn_mask"]
        b_rec = self.initializer["b_rec"]
        b_out = self.initializer["b_out"]
        w_out = self.initializer["w_out"]
        taus_gaus = self.initializer["taus_gaus"]

        if settings["rand_init_x"]:
            init_state = np.random.normal(0, 0.1, size=[self.N, batch_size])
        else:
            init_state = self.initializer["init_state"]
        if "x2x" in settings:
            if settings["x2x"]:
                mult = 2.0
        else:
            mult = 1.0

        # Synaptic currents and firing-rates
        x = np.zeros((T, self.N, batch_size))  # synaptic currents
        r = np.zeros((T, self.N, batch_size))  # firing-rates
        o = np.zeros((T, self.out_channels, batch_size))

        # activation functions
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        relu = lambda x: np.maximum(x, 0)
        softmax = (
            lambda x: np.exp(x - np.max(x, axis=1, keepdims=True))
            / np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1)[
                :, np.newaxis, :
            ]
        )
        if self.activation == "sigmoid":
            transfer_function = sigmoid
        elif self.activation == "relu":
            transfer_function = relu
        elif self.activation == "tanh":
            transfer_function = lambda x: np.tanh(x)

        # initialize numpy arrays for forward pass
        x[0] = init_state
        r[0] = transfer_function(x[0])

        o = np.zeros((T, self.out_channels, batch_size))
   
        # Pass the synaptic time constants thru the sigmoid function
        if len(tau_lims) > 1:
            taus_sig = sigmoid(taus_gaus) * (tau_lims[1] - tau_lims[0]) + tau_lims[0]
        elif len(tau_lims) == 1:  # one scalar synaptic decay time-constant
            taus_sig = tau_lims[0]
        alpha = deltaT / taus_sig

        # apply dales law and conn masks
        if self.apply_dale:
            w_rec = relu(w_rec)
        ww = np.matmul(w_rec, dale_mask) * conn_mask
        ww_in = np.multiply(w_in_scale, w_in)

        for t in range(1, T):
            r_post = r[t - 1]

            # recurrent layer update
            x[t] = (
                ((1 - alpha) * x[t - 1])
                + (
                    alpha
                    * (
                        (np.matmul(ww, r_post) + b_rec)
                        + np.matmul(ww_in, u[:, :, t - 1])
                    )
                )
                + np.sqrt(2.0 * alpha * self.rec_noise * self.rec_noise)
                * np.random.randn(self.N, batch_size)
            )
            r[t] = transfer_function(x[t] * mult)

            # output
            o[t] = np.matmul(w_out, r[t]) + b_out
        if settings["loss"] == "sce":
            o = softmax(o)
        return x, r, o


def set_gpu(gpu, frac):
    """
    Specify which and how much GPU to use:
    Args:
        gpu: String telling which gpu to use
        gpu_frac: scalar in [0,1], how much of the simmulation to use on gpu.
    Returns:
        tensorflow config
    """

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=frac)
    return gpu_options


def lfp_p(x, t_conn_mask, t_dale_mask, t_w_rec, dt,T, activation):
    """
    Calculate RNN's highest power LFP frequency

    Args:
        x: RNN hidden states
        t_conn_mask: recurrent connectivity mask
        t_dale_mask: recurrent dale mask
        t_w_rec: recurrent weights
        dt: model timestep
        T: trial duration
        activation: string denotion model activation function
    Returns:
        freq: frequency with highest LFP power
    """
    # activation functions
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    relu = lambda x: np.maximum(x, 0)
   
    if activation == "sigmoid":
        transfer_function = sigmoid
    elif activation == "relu":
        transfer_function = relu
    elif activation == "tanh":
        transfer_function = lambda x: np.tanh(x)


    freqs_l = np.logspace(*np.log10([0.1, 30]), num=50)
    tim = np.arange(T) * dt / 1000

    # calculate effective recurrent weights
    if np.min(t_dale_mask)<0:
        ww = relu(t_w_rec)
        ww = np.matmul(ww, t_dale_mask)
    else:
        ww = t_w_rec
    ww = np.multiply(ww,t_conn_mask)

    LFP = []
    #calculate LFP
    for xs in x:
        LFP.append(np.matmul(np.abs(ww), transfer_function(xs)))
    LFP = np.array(LFP)
    LFP = np.mean(LFP, axis=1)
    _, amp = scalogram(
        LFP[:, 0],
        7,
        tim,
        dt / 1000,
        freqs_l,
        #normalize=True,
    )
    #calculate time frequency representation
    for tr in range(1, np.shape(LFP)[1]):
        _, amp_tr = scalogram(
            LFP[:, tr],
            7,
            tim,
            dt / 1000,
            freqs_l,
            #normalize=True,
        )
        amp += amp_tr

    #find highest power
    freq = freqs_l[np.argmax(np.mean(amp, axis=1))]

    return freq
