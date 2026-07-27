import logging
import collections

import numpy as np
import torch
from torch import nn

from . import dqn_agent
from ..tools import flag_tools


class DqnReprAgent(dqn_agent.DqnAgent):

    def __init__(self, reward_mode='sparse', dist_reward_coeff=1.0, 
                 goal_obs_prepro=None, repr_loss_weight=1.0, discounts = None, repr_model_factory = None, **kwargs):
        assert reward_mode in ['sparse', 'l2', 'mix', 'rawmix']
        self._reward_mode = reward_mode
        self._dist_reward_coeff = dist_reward_coeff
        self._goal_obs_prepro = goal_obs_prepro
        self._repr_loss_weight = repr_loss_weight  # How much to prioritize map-making
        self._discounts = [0.1, 0.9]
        self._repr_model_factory = repr_model_factory
        super().__init__(**kwargs)

    def _build_model(self):
        super()._build_model()
        # Extract the actual model instances from the wrapper
        self._repr_fn_short = self._model.repr_fn_short
        self._repr_fn_long = self._model.repr_fn_long
    
        # Combined optimizer for the "Dual-Discount" brain
        self._repr_optimizer = torch.optim.Adam(
            list(self._repr_fn_short.parameters()) + 
            list(self._repr_fn_long.parameters()),
            lr=self._learning_rate
        )

    def _get_train_batch(self):
        # 1. Get standard transitions for DQN (s1, s2, a, r, dsc)
        steps1, steps2 = self._replay_buffer.sample_transitions(batch_size=self._batch_size)
        
        # 2. Get Dual-Discount pairs (Short: 0.1, Long: 0.99)
        # s_p[0] is s1_short, s_p[1] is s1_long
        s_p, ns_p = self._replay_buffer.sample_pairs(
                self._batch_size, 
                discount=self._discounts)
        
        # 3. Get Negative samples to prevent collapse
        neg_steps = self._replay_buffer.sample_steps(self._batch_size)

        # initiallize the batch container
        batch = flag_tools.Flags()

        # Standard DQN data
        batch.s1 = self._tensor(self._get_obs_batch(steps1))
        batch.s2 = self._tensor(self._get_obs_batch(steps2))
        batch.a = self._tensor(self._get_action_batch(steps1))
        batch.r, batch.dsc = map(self._tensor, self._get_r_dsc_batch(steps2))
        batch.sg = self._tensor(self._get_goal_obs_batch(steps2))

        # Dual-Model Representation data
        batch.s1_short = self._tensor(self._get_obs_batch(s_p[0]))
        batch.s2_short = self._tensor(self._get_obs_batch(ns_p[0]))
        batch.s1_long = self._tensor(self._get_obs_batch(s_p[1]))
        batch.s2_long = self._tensor(self._get_obs_batch(ns_p[1]))

        # Negative samples obs
        batch.s_neg = self._tensor(self._get_obs_batch(neg_steps))

        # Update reward with the learned map (Mix Mode)
        batch.r = self._get_repr_reward(batch.r, batch.s2, batch.sg)
        return batch

    def _get_repr_reward(self, r, s2, sg):
        if self._reward_mode == 'l2':
            # Compute raw Euclidean distance (no learned model)
            dist = (s2 - sg).flatten(start_dim=1).norm(dim=-1)
            r = - dist * self._dist_reward_coeff
            
        elif self._reward_mode == 'rawmix':
            # Mix raw Euclidean distance with original reward
            dist = (s2 - sg).flatten(start_dim=1).norm(dim=-1)
            r = - dist * self._dist_reward_coeff * 0.5 + r * 0.5
            
        elif self._reward_mode == 'mix':
            # The NEW learned Dual-Discount logic
            with torch.no_grad():
                # Concatenate short and long for a "multi-scale" distance
                s2_rep = torch.cat([
                    self._repr_fn_short(s2), 
                    self._repr_fn_long(s2)
                ], dim=-1)
                sg_rep = torch.cat([
                    self._repr_fn_short(sg), 
                    self._repr_fn_long(sg)
                ], dim=-1)
                
            dist = (s2_rep - sg_rep).norm(dim=-1)
            r = - dist * self._dist_reward_coeff * 0.5 + r * 0.5
            
    # If reward_mode == 'sparse', skip the 'if/elif' blocks 
    # and return the original 'r' as provided by the environment.
        return r

    def _train_step(self, batch):
        # Update DQN
        dqn_loss = self._compute_dqn_loss(batch)
        self._q_optimizer.zero_grad()
        dqn_loss.backward()
        self._q_optimizer.step()

        # Update Representations (The Laplacian logic from train_laprepr.py)
        self._repr_optimizer.zero_grad()
        
        phi_s1_s = self._repr_fn_short(batch.s1_short)
        phi_s2_s = self._repr_fn_short(batch.s2_short)
        phi_s1_l = self._repr_fn_long(batch.s1_long)
        phi_s2_l = self._repr_fn_long(batch.s2_long)
        phi_neg_s  = self._repr_fn_short(batch.s_neg)
        phi_neg_l  = self._repr_fn_long(batch.s_neg)

        # 1. Positive Loss: Minimize distance between neighbors
        loss_pos = torch.mean(torch.sum((phi_s1_s - phi_s2_s)**2, dim=-1)) + \
                   torch.mean(torch.sum((phi_s1_l - phi_s2_l)**2, dim=-1))
        
        # 2. Negative Loss: Maximize distance between random states 
        # Using a simple repulsive hinge loss or exponential
        loss_neg = torch.mean(torch.exp(-torch.norm(phi_s1_s - phi_neg_s, p=2, dim=-1))) + \
                torch.mean(torch.exp(-torch.norm(phi_s1_l - phi_neg_l, p=2, dim=-1)))

        total_repr_loss = (loss_pos + loss_neg) * self._repr_loss_weight
        total_repr_loss.backward()
        self._repr_optimizer.step()

        return dqn_loss.item(), total_repr_loss.item()
            

class DqnReprAgentModel(nn.Module):

    def __init__(
            self, 
            q_model_factory, 
            repr_model_factory,
            ):
        super().__init__()
        self.q_fn_learning = q_model_factory()
        self.q_fn_target = q_model_factory()
        # create two distinct instance of repr
        self.repr_fn_long = repr_model_factory() # Call 1: Creates one CNN
        self.repr_fn_short = repr_model_factory() # Call 2: Creates a second, identical CNN


class DqnReprAgentConfig(dqn_agent.DqnAgentConfig):

    def _set_default_flags(self):
        super()._set_default_flags()
        flags = self._flags
        flags.reward_mode = 'sparse'
        flags.dist_reward_coeff = 1.0
        flags.repr_model_cfg = None

    def _repr_model_factory(self):
        """Build model and load from checkpoint."""
        raise NotImplementedError

    def _goal_obs_prepro(self, obs):
        """Get preprocessed goal representation."""
        raise NotImplementedError

    def _model_factory(self):
        return DqnReprAgentModel(
                q_model_factory=self._q_model_factory,
                repr_model_factory=self._repr_model_factory
                )

    def _build_args(self):
        super()._build_args()
        args = self._args
        flags = self._flags
        args.reward_mode = flags.reward_mode
        args.dist_reward_coeff = flags.dist_reward_coeff
        args.goal_obs_prepro = self._goal_obs_prepro
        args.repr_loss_weight = getattr(flags, 'repr_loss_weight', 1.0)

















