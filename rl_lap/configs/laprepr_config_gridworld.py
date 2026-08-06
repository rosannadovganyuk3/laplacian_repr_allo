import numpy as np
from ..agent import laprepr
from ..envs.gridworld import gridworld_envs
from . import networks
import torch

class Config(laprepr.LapReprConfig):

    def _set_default_flags(self):
        super()._set_default_flags()
        flags = self._flags
        if torch.cuda.is_available():
            flags.device = 'cuda'
        elif getattr(torch.backends, 'mps', None) is not None \
                and torch.backends.mps.is_available():
            flags.device = 'mps'
        else:
            flags.device = 'cpu'
        # add layers and units for CNN
        flags.n_layers = 3
        flags.n_units = 256
        flags.rnn_type = 'LSTM' # 'LSTM', 'GRU', or 'RNN'
        flags.d = 20
        flags.n_samples = 200000
        flags.batch_size = 256 # changed from 128
        flags.discount = [0.1, 0.9] # [short-term, long-term]
        flags.w_neg = 15.0 # changed from 1.0
        flags.c_neg = 2.0 # changed from 1.0
        flags.reg_neg = 0.1 #changed from 0 to 0.001 (value for neg_loss reg)
        flags.reg_start_step = 0 # regularization start threshold
        flags.lagrange_mult= 0.1 # changed from 1.0 to 0.1 -> 0.9 to 0.001
        flags.replay_buffer_size = 200000
        flags.opt_args.name = 'Adam'
        flags.opt_args.lr = 0.0001 # changed from 0.0003
        # train
        flags.log_dir = '/tmp/rl_laprepr/log'
        flags.total_train_steps = 200000 # changed from 30000 to 50000
        flags.print_freq = 1000
        flags.save_freq = 10000

    def _obs_prepro(self, obs):
        """
        Processes the 5x5x3 image for the CNN.
        1. Extract the image.
        2. Transpose from (H, W, C) to (C, H, W) for PyTorch.
        pos_to_obs already builds the image from colour constants in [0, 1],
        so no further normalization is needed here.
        """
        img = obs.agent.image # Access the 5x5x3 grid we created
        # PyTorch Conv2d expects (Channels, Height, Width)
        img = np.transpose(img, (2, 0, 1))
        return img.astype(np.float32)

    def _env_factory(self):
        return gridworld_envs.make(self._flags.env_id)

    def _model_factory(self):
        # Swap ReprNetMLP for your new ReprNetRNN
        return networks.ReprNetRNN(
                input_shape=self._obs_shape,
                n_layers=self._flags.n_layers,
                n_units=self._flags.n_units,
                d=self._flags.d,
                rnn_type=self._flags.rnn_type)


