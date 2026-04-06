import os
import logging
import torch

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
        # Gridworld specific: use agent position as the state
        return obs.agent.position

    def _goal_obs_prepro(self, obs):
        # Gridworld specific: use goal position
        return obs.goal.position

    def _env_factory(self):
        return gridworld_envs.make(self._flags.env_id)

    def _q_model_factory(self):
        return networks.DiscreteQNetMLP(
                input_shape=self._obs_shape, 
                n_actions=self._action_spec.n, 
                n_layers=3, 
                n_units=256)

    def _repr_model_factory(self):
        """Returns a factory function that creates a fresh Laplacian MLP."""
        # We return a lambda so the agent can call it twice: 
        # once for 'short' and once for 'long'.
        from ..agent import laprepr
        return lambda: laprepr.LapReprMLP(
            obs_spec=self._obs_shape, 
            d=self._flags.repr_dim)

    

    def _build_args(self):
        # Ensure the new flags are passed into the Agent's __init__
        super()._build_args()
        self._args.repr_loss_weight = self._flags.repr_loss_weight
        self._args.discounts = self._flags.discounts