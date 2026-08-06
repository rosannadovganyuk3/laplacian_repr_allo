import os
import logging
import torch
import numpy as np
from ..agent import dqn_repr_agent
from ..envs.gridworld import gridworld_envs
from . import networks
# We import the laprepr network directly instead of the pretrained config
# Becuase switching from Offline to Onlne learning 
# laprepr_config had pre-trained model (so removed)
# need agent to build new networks from laprepr architecture
from ..agent import laprepr 
from ..tools import flag_tools


class Config(dqn_repr_agent.DqnReprAgentConfig):

    def _set_default_flags(self):
        super()._set_default_flags()
        flags = self._flags
        # Standard DQN hyperparams
        flags.batch_size = 128
        flags.discount = 0.98
        flags.update_freq = 50
        flags.update_rate = 0.05
        flags.opt_args.name = 'Adam'
        flags.opt_args.lr = 0.001
        
        # Replay Buffer settings
        flags.replay_buffer_init = 10000
        flags.replay_buffer_size = int(1e6)
        
        # Dual-Discount Representation hyperparams
        flags.repr_dim = 16          # Size of the learned embedding
        flags.repr_loss_weight = 1.0  # Balance between DQN and Laplacian loss
        flags.dist_reward_coeff = 1.0
        flags.reward_mode = 'mix'

        flags.discounts = [0.1, 0.9]
        flags.repr_dim = 16 #lapreprlmp knows output size

    def _obs_prepro(self, obs):
        # Must match the others: (H,W,C) -> (C,H,W).
        # pos_to_obs already builds the image from colour constants in [0, 1],
        # so no further normalization is needed here.
        img = obs.agent.image
        img = np.transpose(img, (2, 0, 1))
        return img.astype(np.float32)

    def _goal_obs_prepro(self, obs):
        # If your goal also provides an image/grid representation:
        img = obs.goal.image
        img = np.transpose(img, (2, 0, 1))
        return img.astype(np.float32)
    
    def _env_factory(self):
        return gridworld_envs.make(self._flags.env_id)

    def _q_model_factory(self):
        # Swap DiscreteQNetMLP for DiscreteQNetCNN
        return networks.DiscreteQNetRNN(
                input_shape=self._obs_shape, # This will now be (3, 5, 5)
                n_actions=self._action_spec.n, 
                n_layers=self._flags.n_layers, 
                n_units=self._flags.n_units)

    def _repr_model_factory(self):
        """Returns a factory function that creates a fresh Laplacian CNN."""
        # We return a lambda so the agent can call it twice: 
        # once for 'short' and once for 'long';
        # lets the agent create two identical CNNs.
        return lambda: networks.ReprNetCNN(
            input_shape=self._obs_shape, # Matches  networks.py
            n_layers=self._flags.n_layers, 
            n_units=self._flags.n_units,   
            d=self._flags.repr_dim)    

    

    def _build_args(self):
        # Ensure the new flags are passed into the Agent's __init__
        super()._build_args()
        self._args.repr_loss_weight = self._flags.repr_loss_weight
        self._args.discounts = self._flags.discounts