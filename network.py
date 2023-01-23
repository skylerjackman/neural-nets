import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
#added 1/17 for plotting output after each validation step
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #fix for weird issue where matplotlib kills  kernel


class DMTSNet(pl.LightningModule):
    """DelayedMatchToSampleNetwork. Class defines RNN for solving a DMTS task. """
    #default
    def __init__(
        self,
        rnn_type,
        input_size,
        hidden_size,
        output_size,
        dt_ann,
        alpha,
        alpha_W,
        g,
        nl,
        lr,
    ):
        super().__init__()
        self.stsp = False
        self.dt_ann = dt_ann
        self.lr = lr
        self.save_hyperparameters()
        self.act_reg = 0
        self.param_reg = 0
        
        
        if rnn_type == "fixed":
            # if model is fixed: excitatory + inhibitory hidden nodes
            self.rnn = fixRNNLayer(input_size, hidden_size, output_size, alpha, dt_ann, g, nl) #dt_ann is only used for x and u, but keeping it here for consistency
            self.fixed_syn = True

        if rnn_type == "stsp":
            # if model is Mongillo/Masse STSP: excitatory + inhibitory hidden nodes AND x+u update W during trial
            self.rnn = stspRNNLayer(input_size, hidden_size, output_size, alpha, dt_ann, g, nl)
            self.fixed_syn = False
            self.stsp = True
            
        if rnn_type == "vanilla":
            #no excitatory or inhibitory stuff; just a vanilla netowrk
            self.rnn = vRNNLayer(input_size, hidden_size, output_size, alpha, g, nl)
            self.fixed_syn = True
            
        #1/17 addition
        if rnn_type == "stsp_no_fac":
            #Mongillo but no facilitation
            self.rnn = stspnofacRNNLayer(input_size, hidden_size, output_size, alpha, dt_ann, g, nl)
            self.fixed_syn = False
            self.stsp = True
        
    def forward(self, x):
        # defines foward method using the chosen RNN type
        out_readout, out_hidden, w_hidden, _ = self.rnn(x)
        return out_readout, out_hidden, w_hidden, _
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        inp, out_des, y, test_on = batch
        out_readout, out_hidden, _, _ = self.rnn(inp)

        #accumulate losses. if penalizing activity, then add it to the loss
        if self.act_reg != 0:
            reg_loss = self.act_reg*out_hidden.norm(p = 'fro')
            reg_loss /= out_hidden.shape[0]*out_hidden.shape[1]*out_hidden.shape[2]
        else:
            reg_loss = 0
            
        self.log("train_loss_reg", reg_loss) #loss from regularization
        
        ## loss calc'd across entire trial
        # target_loss = F.mse_loss(out_readout, out_des)
        
        ## loss modulatd with respect to certain time intervals
        loss_mask = torch.tensor(np.ones(out_readout.size()))
        samp_on = 67 #1000 in raw samples
        grace = int(100/15) #first 100ms of sample window; divide by dt=15

        for i in range(out_readout.size()[0]): #for each trial

            loss_mask[samp_on:samp_on+grace] = 0 #grace period at start of sample

            delay_on = int(test_on[i]) #don't know why but this is needed or indexing breaks
            delay_off = delay_on + 33

            loss_mask[i, delay_on:delay_on+grace, :] = 0 #grace period at start of test
            loss_mask[i, delay_on+grace:delay_off, :] = 5 #respose window is important

        #loss_mask[:,:,2] *= 2 # if doubling fixation cost
        
        target_loss = torch.mean((out_readout - out_des)**2 * loss_mask.to(self.device))
        
        ## loss calc'd only for test window
        # target_loss = 0
        # for i in test_on.unique():
        #     inds = torch.where(test_on == i)[0]
        #     test_end = int(i) + int(500 / self.dt_ann)
        #     response_end = test_end + int(500 / self.dt_ann)
        #     target_loss += F.mse_loss(
        #         out_readout[inds, test_end:response_end],
        #         out_des[inds, test_end:response_end],
        #     )
            
        # added from validation step- get correct vs incorrect trials to log for training
        accs = np.zeros(out_readout.shape[0])
        for i in range(out_readout.shape[0]):
            curr_max = (
                out_readout[i, int(test_on[i])+int(500/self.dt_ann):int(test_on[i])+2*int(500/self.dt_ann), :-1]
                .argmax(dim=1).cpu().detach().numpy()
            )
            accs[i] = (y[i].item() == curr_max).sum() / len(curr_max)

        self.log("train_acc", accs.mean())
        self.log("train_loss_target", target_loss) #loss from matching targets
        
        loss = target_loss + reg_loss
        self.log("train_loss_total", loss) #total loss
 
        return loss
    
    def validation_step(self, batch, batch_idx):
        # defines validation step
        inp, out_des, y, test_on = batch
        out_readout, _, _, _ = self.rnn(inp)

        accs = np.zeros(out_readout.shape[0])
        # test model performance
        for i in range(out_readout.shape[0]):
            curr_max = (
                out_readout[
                    i,
                    int(test_on[i])
                    + int(500 / self.dt_ann) : int(test_on[i])
                    + 2 * int(500 / self.dt_ann),
                    :-1,
                ]
                .argmax(dim=1)
                .cpu()
                .detach()
                .numpy()
            )
            accs[i] = (y[i].item() == curr_max).sum() / len(curr_max)
            
        ## val loss for entire trial, not just test window
        total_loss = F.mse_loss(out_readout, out_des)
            
        # added from training step- get test window loss to log for each epoch
        loss = 0
        for i in test_on.unique():
            inds = torch.where(test_on == i)[0]
            test_end = int(i) + int(500 / self.dt_ann)
            response_end = test_end + int(500 / self.dt_ann)
            loss += F.mse_loss(out_readout[inds, test_end:response_end], out_des[inds, test_end:response_end])
            
        self.log("val_test_loss", loss)
        self.log("val_total_loss", total_loss)
        self.log("val_acc", accs.mean(), prog_bar=True)
        
        self.log("mean_hidden_weight", self.rnn.W.mean()) #mean for all hidden-layer weights
        
        #1/17 edit: plot NN output for each trial type
        f,ax=plt.subplots(2,5, figsize=(20,5))
        with torch.no_grad():
            for sample in range(2): #2 sample inputs
                for i,delay in enumerate(torch.unique(test_on)): #for each delay
                    inds = torch.where((y == sample) & (test_on == delay))[0]
                    ax[sample,i].set_ylim(-.1,1.1)
                    ax[sample,i].plot(out_readout[inds].cpu().mean(0))
        plt.savefig("training_output.png")


    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        # by default, we use an L2 weight decay on all parameters.
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.param_reg)

        # lr_scheduler = {'scheduler':  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1),"monitor": 'val_acc'}
        return [optimizer]  # ,[lr_scheduler]
    
    
    
    
class fixRNNLayer(pl.LightningModule):
    """No Mongillo/Masse pre-synaptic STSP"""
    
    def __init__(self, input_size, hidden_size, output_size, alpha, dt, g, nonlinearity):
        super(fixRNNLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha
        self.inv_sqrt_alpha = 1 / np.sqrt(alpha)
        self.root_inv_hidden = 1 / np.sqrt(hidden_size)
        self.g = g
        self.dt = dt
        self.f_out = nn.Softplus()
        self.process_noise = 0.05

        # no time-constants for the network

        # define nonlinearity for the neural dynamics
        self.nonlinearity = nonlinearity
        if nonlinearity == "tanh":
            self.phi = torch.tanh
        if nonlinearity == "relu":
            self.phi = F.relu
        if nonlinearity == "retanh":
            self.phi = torch.nn.ReLU(torch.nn.Tanh())
        if nonlinearity == "none":
            self.phi = torch.nn.Identity()

        # initialize input-to-hidden weights
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(hidden_size, input_size).uniform_(
                -self.root_inv_hidden, self.root_inv_hidden
            )
        )

        # initialize hidden to output weights
        self.weight_ho = nn.Parameter(
            torch.FloatTensor(output_size, hidden_size).uniform_(
                -self.root_inv_hidden, self.root_inv_hidden
            )
        )

        # initialize hidden-to-hidden weights
        W = torch.FloatTensor(hidden_size, hidden_size).uniform_(
            -self.root_inv_hidden, self.root_inv_hidden
        )
        W.log_normal_(0, self.g / np.sqrt(hidden_size))
        W = F.relu(W)
        W /= 10 * (torch.linalg.vector_norm(W, ord=2))
        self.W = nn.Parameter(W)

        # define seperate inhibitory and excitatory neural populations
        diag_elements_of_D = torch.ones(self.hidden_size)
        diag_elements_of_D[int(0.8 * self.hidden_size) :] = -1
        syn_inds_rand = torch.randperm(self.hidden_size)
        diag_elements_of_D = diag_elements_of_D[syn_inds_rand]
        D = diag_elements_of_D.diag_embed()

        self.register_buffer("D", D)

        # no facil + depress buffers

        # no time constant buffers

        # initialize output bias
        self.bias_oh = nn.Parameter(
            0 * torch.normal(0, 1 / np.sqrt(hidden_size), (1, output_size))
        )

        # initialize hidden bias
        self.bias_hh = nn.Parameter(
            0 * torch.normal(0, 1 / np.sqrt(hidden_size), (1, hidden_size))
        )

        # for structurally perturbing weight matrix
        self.struc_p_0 = 0
        self.register_buffer(
            "struc_perturb_mask",
            torch.FloatTensor(self.hidden_size, self.hidden_size).uniform_()
            > self.struc_p_0,
        )

    def forward(self, input):

        # initialize neural state; no synaptic states (x,u)
        state = 0 * torch.randn(input.shape[0], self.hidden_size, device=self.device)

        # defines process noise
        noise = (
            1.41
            * self.process_noise
            * torch.randn(
                input.shape[0], input.shape[1], self.hidden_size, device=self.device
            )
        )

        # for storing neural outputs + hidden states; no synaptic states
        outputs = []
        states = []

        for i in range(input.shape[1]):

            # compute and save neural output
            hy = state @ self.weight_ho.T + self.bias_oh
            outputs += [hy]

            # save neural hidden states; no synaptic states
            states += [state]

            # no update for synaptic variables

            # presynaptic input not modulated based on STSP rule; just apply E/I buffer and rectify weights
            I = state @ (
                (self.D @ F.relu(self.W)) * self.struc_perturb_mask
            )

            # compute neural update
            fstate = -state + self.phi(
                I
                + input[:, i, :] @ self.weight_ih.T
                + self.bias_hh
                + self.inv_sqrt_alpha * noise[:, i, :]
            )

            # step neural states forward; no synaptic update
            state = state + self.alpha * fstate

        # no synaptic variables

        return (
            torch.stack(outputs).permute(1, 0, 2),
            torch.stack(states).permute(1, 0, 2),
            None, #where x and u would go (w_hidden), return None
            noise,
        )

    

class stspRNNLayer(pl.LightningModule):
    """Implements the RNN of Mongillo/Masse, using pre-synaptic STSP"""

    def __init__(self, input_size, hidden_size, output_size, alpha, dt, g, nonlinearity):
        super(stspRNNLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha
        self.inv_sqrt_alpha = 1 / np.sqrt(alpha)
        self.root_inv_hidden = 1 / np.sqrt(hidden_size)
        self.g = g
        self.dt = dt
        self.f_out = nn.Softplus()
        self.process_noise = 0.05

        # define time-constants for the network, in units of ms
        self.tau_x_facil = 200
        self.tau_u_facil = 1500
        self.U_facil = 0.15

        self.tau_x_depress = 1500
        self.tau_u_depress = 200
        self.U_depress = 0.45

        # define nonlinearity for the neural dynamics
        self.nonlinearity = nonlinearity
        if nonlinearity == "tanh":
            self.phi = torch.tanh
        if nonlinearity == "relu":
            self.phi = F.relu
        if nonlinearity == "retanh":
            self.phi = torch.nn.ReLU(torch.nn.Tanh())
        if nonlinearity == "none":
            self.phi = torch.nn.Identity()

        # initialize input-to-hidden weights
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(hidden_size, input_size).uniform_(
                -self.root_inv_hidden, self.root_inv_hidden
            )
        )

        # initialize hidden to output weights
        self.weight_ho = nn.Parameter(
            torch.FloatTensor(output_size, hidden_size).uniform_(
                -self.root_inv_hidden, self.root_inv_hidden
            )
        )

        # initialize hidden-to-hidden weights
        W = torch.FloatTensor(hidden_size, hidden_size).uniform_(
            -self.root_inv_hidden, self.root_inv_hidden
        )
        W.log_normal_(0, self.g / np.sqrt(hidden_size))
        W = F.relu(W)
        W /= 10 * (torch.linalg.vector_norm(W, ord=2))
        self.W = nn.Parameter(W)

        # define seperate inhibitory and excitatory neural populations
        diag_elements_of_D = torch.ones(self.hidden_size)
        diag_elements_of_D[int(0.8 * self.hidden_size) :] = -1
        syn_inds = torch.arange(self.hidden_size)
        syn_inds_rand = torch.randperm(self.hidden_size)
        diag_elements_of_D = diag_elements_of_D[syn_inds_rand]
        D = diag_elements_of_D.diag_embed()

        self.register_buffer("D", D)

        self.register_buffer("facil_syn_inds", syn_inds[: int(self.hidden_size / 2)])
        self.register_buffer("depress_syn_inds", syn_inds[int(self.hidden_size / 2) :])

        # time constants
        tau_x = torch.ones(self.hidden_size)
        tau_x[self.facil_syn_inds] = self.tau_x_facil
        tau_x[self.depress_syn_inds] = self.tau_x_depress
        self.register_buffer("Tau_x", (1 / tau_x))

        tau_u = torch.ones(self.hidden_size)
        tau_u[self.facil_syn_inds] = self.tau_u_facil
        tau_u[self.depress_syn_inds] = self.tau_u_depress
        self.register_buffer("Tau_u", (1 / tau_x))

        U = torch.ones(self.hidden_size)
        U[self.facil_syn_inds] = self.U_facil
        U[self.depress_syn_inds] = self.U_depress
        self.register_buffer("U", U)

        # initialize output bias
        self.bias_oh = nn.Parameter(
            0 * torch.normal(0, 1 / np.sqrt(hidden_size), (1, output_size))
        )

        # initialize hidden bias
        self.bias_hh = nn.Parameter(
            0 * torch.normal(0, 1 / np.sqrt(hidden_size), (1, hidden_size))
        )

        # for structurally perturbing weight matrix
        self.struc_p_0 = 0
        self.register_buffer(
            "struc_perturb_mask",
            torch.FloatTensor(self.hidden_size, self.hidden_size).uniform_()
            > self.struc_p_0,
        )

    def forward(self, input):

        # initialize neural state and synaptic states
        state = 0 * torch.randn(input.shape[0], self.hidden_size, device=self.device)
        u_state = 0 * torch.rand(input.shape[0], self.hidden_size, device=self.device)
        x_state = torch.ones(input.shape[0], self.hidden_size, device=self.device)

        # defines process noise
        noise = (
            1.41
            * self.process_noise
            * torch.randn(
                input.shape[0], input.shape[1], self.hidden_size, device=self.device
            )
        )

        # for storing neural outputs, hidden states, and synaptic states
        outputs = []
        states = []
        states_x = []
        states_u = []

        for i in range(input.shape[1]):

            # compute and save neural output
            hy = state @ self.weight_ho.T + self.bias_oh
            outputs += [hy]

            # save neural and synaptic hidden states
            states += [state]
            states_x += [x_state]
            states_u += [u_state]

            # compute update for synaptic variables
            fx = (1 - x_state) * self.Tau_x - u_state * x_state * state * (
                self.dt / 1000
            )
            fu = (self.U - u_state) * self.Tau_u + self.U * (1 - u_state) * state * (
                self.dt / 1000
            )

            # define modulated presynaptic input based on STSP rule
            I = (x_state * state * u_state) @ (
                (self.D @ F.relu(self.W)) * self.struc_perturb_mask
            )

            # compute neural update
            fstate = -state + self.phi(
                I
                + input[:, i, :] @ self.weight_ih.T
                + self.bias_hh
                + self.inv_sqrt_alpha * noise[:, i, :]
            )

            # step neural and synaptic states forward
            state = state + self.alpha * fstate
            x_state = torch.clamp(x_state + self.alpha * fx, min=0, max=1)
            u_state = torch.clamp(u_state + self.alpha * fu, min=0, max=1)

        # organize and return variables
        x_hidden = torch.stack(states_x).permute(1, 0, 2)
        u_hidden = torch.stack(states_u).permute(1, 0, 2)

        return (
            torch.stack(outputs).permute(1, 0, 2),
            torch.stack(states).permute(1, 0, 2),
            torch.cat((x_hidden, u_hidden), dim=2),
            noise,
        )

    
class vRNNLayer(pl.LightningModule):
    """Vanilla RNN layer in continuous time."""

    def __init__(self, input_size, hidden_size, output_size, alpha, g, nonlinearity):
        super(vRNNLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha
        self.inv_sqrt_alpha = 1 / np.sqrt(alpha)
        self.cont_stab = False
        self.disc_stab = True
        self.g = g
        self.process_noise = 0.05

        # set nonlinearity of the vRNN
        self.nonlinearity = nonlinearity
        if nonlinearity == "tanh":
            self.phi = torch.tanh
        if nonlinearity == "relu":
            self.phi = F.relu
        if nonlinearity == "none":
            print("Nl = none")
            self.phi = torch.nn.Identity()

        # initialize the input-to-hidden weights
        self.weight_ih = nn.Parameter(
            torch.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, input_size))
        )

        # initialize the hidden-to-output weights
        self.weight_ho = nn.Parameter(
            torch.normal(0, 1 / np.sqrt(hidden_size), (output_size, hidden_size))
        )

        # initialize the hidden-to-hidden weights
        self.W = nn.Parameter(
            torch.normal(0, self.g / np.sqrt(hidden_size), (hidden_size, hidden_size))
        )

        # initialize the output bias weights
        self.bias_oh = nn.Parameter(
            torch.normal(0, 1 / np.sqrt(hidden_size), (1, output_size))
        )

        # initialize the hidden bias weights
        self.bias_hh = nn.Parameter(
            torch.normal(0, 1 / np.sqrt(hidden_size), (1, hidden_size))
        )

        # define mask for weight matrix do to structural perturbation experiments
        self.struc_p_0 = 0
        self.register_buffer(
            "struc_perturb_mask",
            torch.FloatTensor(self.hidden_size, self.hidden_size).uniform_()
            > self.struc_p_0,
        )

    def forward(self, input):

        # initialize state at the origin. randn is there just in case we want to play with this later.
        state = 0 * torch.randn(input.shape[0], self.hidden_size, device=self.device)

        # defines process noise using Euler-discretization of stochastic differential equation defining the RNN
        noise = (
            1.41
            * self.process_noise
            * torch.randn(
                input.shape[0], input.shape[1], self.hidden_size, device=self.device
            )
        )

        # for storing RNN outputs and hidden states
        outputs = []
        states = []

        # loop over input
        for i in range(input.shape[1]):

            # compute output
            hy = state @ self.weight_ho.T + self.bias_oh

            # save output and hidden states
            outputs += [hy]
            states += [state]

            # compute the RNN update
            fx = -state + self.phi(
                state @ (self.W * self.struc_perturb_mask)
                + input[:, i, :] @ self.weight_ih.T
                + self.bias_hh
                + self.inv_sqrt_alpha * noise[:, i, :]
            )

            # step hidden state foward using Euler discretization
            state = state + self.alpha * (fx)

        # organize states and outputs and return
        return (
            torch.stack(outputs).permute(1, 0, 2),
            torch.stack(states).permute(1, 0, 2),
            noise,
            None,
        )
    
    
    
class stspnofacRNNLayer(pl.LightningModule):
    """Implements the RNN of Mongillo/Masse, using pre-synaptic STSP"""

    def __init__(self, input_size, hidden_size, output_size, alpha, dt, g, nonlinearity):
        super(stspRNNLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha
        self.inv_sqrt_alpha = 1 / np.sqrt(alpha)
        self.root_inv_hidden = 1 / np.sqrt(hidden_size)
        self.g = g
        self.dt = dt
        self.f_out = nn.Softplus()
        self.process_noise = 0.05

        # define time-constants for the network, in units of ms
        self.tau_x_facil = 200
        self.tau_u_facil = 0
        self.U_facil = 0.15

        self.tau_x_depress = 1500
        self.tau_u_depress = 200
        self.U_depress = 0.45

        # define nonlinearity for the neural dynamics
        self.nonlinearity = nonlinearity
        if nonlinearity == "tanh":
            self.phi = torch.tanh
        if nonlinearity == "relu":
            self.phi = F.relu
        if nonlinearity == "retanh":
            self.phi = torch.nn.ReLU(torch.nn.Tanh())
        if nonlinearity == "none":
            self.phi = torch.nn.Identity()

        # initialize input-to-hidden weights
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(hidden_size, input_size).uniform_(
                -self.root_inv_hidden, self.root_inv_hidden
            )
        )

        # initialize hidden to output weights
        self.weight_ho = nn.Parameter(
            torch.FloatTensor(output_size, hidden_size).uniform_(
                -self.root_inv_hidden, self.root_inv_hidden
            )
        )

        # initialize hidden-to-hidden weights
        W = torch.FloatTensor(hidden_size, hidden_size).uniform_(
            -self.root_inv_hidden, self.root_inv_hidden
        )
        W.log_normal_(0, self.g / np.sqrt(hidden_size))
        W = F.relu(W)
        W /= 10 * (torch.linalg.vector_norm(W, ord=2))
        self.W = nn.Parameter(W)

        # define seperate inhibitory and excitatory neural populations
        diag_elements_of_D = torch.ones(self.hidden_size)
        diag_elements_of_D[int(0.8 * self.hidden_size) :] = -1
        syn_inds = torch.arange(self.hidden_size)
        syn_inds_rand = torch.randperm(self.hidden_size)
        diag_elements_of_D = diag_elements_of_D[syn_inds_rand]
        D = diag_elements_of_D.diag_embed()

        self.register_buffer("D", D)

        self.register_buffer("facil_syn_inds", syn_inds[: int(self.hidden_size / 2)])
        self.register_buffer("depress_syn_inds", syn_inds[int(self.hidden_size / 2) :])

        # time constants
        tau_x = torch.ones(self.hidden_size)
        tau_x[self.facil_syn_inds] = self.tau_x_facil
        tau_x[self.depress_syn_inds] = self.tau_x_depress
        self.register_buffer("Tau_x", (1 / tau_x))

        tau_u = torch.ones(self.hidden_size)
        tau_u[self.facil_syn_inds] = self.tau_u_facil
        tau_u[self.depress_syn_inds] = self.tau_u_depress
        self.register_buffer("Tau_u", (1 / tau_x))

        U = torch.ones(self.hidden_size)
        U[self.facil_syn_inds] = self.U_facil
        U[self.depress_syn_inds] = self.U_depress
        self.register_buffer("U", U)

        # initialize output bias
        self.bias_oh = nn.Parameter(
            0 * torch.normal(0, 1 / np.sqrt(hidden_size), (1, output_size))
        )

        # initialize hidden bias
        self.bias_hh = nn.Parameter(
            0 * torch.normal(0, 1 / np.sqrt(hidden_size), (1, hidden_size))
        )

        # for structurally perturbing weight matrix
        self.struc_p_0 = 0
        self.register_buffer(
            "struc_perturb_mask",
            torch.FloatTensor(self.hidden_size, self.hidden_size).uniform_()
            > self.struc_p_0,
        )

    def forward(self, input):

        # initialize neural state and synaptic states
        state = 0 * torch.randn(input.shape[0], self.hidden_size, device=self.device)
        u_state = 0 * torch.rand(input.shape[0], self.hidden_size, device=self.device)
        x_state = torch.ones(input.shape[0], self.hidden_size, device=self.device)

        # defines process noise
        noise = (
            1.41
            * self.process_noise
            * torch.randn(
                input.shape[0], input.shape[1], self.hidden_size, device=self.device
            )
        )

        # for storing neural outputs, hidden states, and synaptic states
        outputs = []
        states = []
        states_x = []
        states_u = []

        for i in range(input.shape[1]):

            # compute and save neural output
            hy = state @ self.weight_ho.T + self.bias_oh
            outputs += [hy]

            # save neural and synaptic hidden states
            states += [state]
            states_x += [x_state]
            states_u += [u_state]

            # compute update for synaptic variables
            fx = (1 - x_state) * self.Tau_x - u_state * x_state * state * (
                self.dt / 1000
            )
            fu = (self.U - u_state) * self.Tau_u + self.U * (1 - u_state) * state * (
                self.dt / 1000
            )

            # define modulated presynaptic input based on STSP rule
            I = (x_state * state * u_state) @ (
                (self.D @ F.relu(self.W)) * self.struc_perturb_mask
            )

            # compute neural update
            fstate = -state + self.phi(
                I
                + input[:, i, :] @ self.weight_ih.T
                + self.bias_hh
                + self.inv_sqrt_alpha * noise[:, i, :]
            )

            # step neural and synaptic states forward
            state = state + self.alpha * fstate
            x_state = torch.clamp(x_state + self.alpha * fx, min=0, max=1)
            u_state = torch.clamp(u_state + self.alpha * fu, min=0, max=1)

        # organize and return variables
        x_hidden = torch.stack(states_x).permute(1, 0, 2)
        u_hidden = torch.stack(states_u).permute(1, 0, 2)

        return (
            torch.stack(outputs).permute(1, 0, 2),
            torch.stack(states).permute(1, 0, 2),
            torch.cat((x_hidden, u_hidden), dim=2),
            noise,
        )
