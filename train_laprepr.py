import os
import argparse
import importlib

from rl_lap.agent import laprepr

from rl_lap.tools import flag_tools
from rl_lap.tools import timer_tools
from rl_lap.tools import logging_tools


parser = argparse.ArgumentParser()
parser.add_argument('--log_base_dir', type=str, 
        default=os.path.join(os.getcwd(), 'log'))
parser.add_argument('--log_sub_dir', type=str, default='test')
parser.add_argument('--env_id', type=str, default='OneRoom')
parser.add_argument('--config_dir', type=str, default='rl_lap.configs')
parser.add_argument('--config_file', 
        type=str, default='laprepr_config_gridworld')
parser.add_argument('--exp_name', type=str, default='laprepr')
parser.add_argument('--args', type=str, action='append', default=[])

FLAGS = parser.parse_args()


def get_config_cls():
    config_module = importlib.import_module(
            FLAGS.config_dir+'.'+FLAGS.config_file)
    config_cls = config_module.Config
    return config_cls

def save_dual_discount_representations(learner, save_path):
    """Save representations from a single batch"""
    batch = learner._get_train_batch()

        # extracts representations from the trained model
        # conver to arrays (.cpu() needed for numpy arrrays)
        # call repr_fn() to extract learned representations
    with torch.no_grad():
        reprs = {
            's1_short': learner._repr_fn(batch.s1_short).cpu().numpy(),
            's2_short': learner._repr_fn(batch.s2_short).cpu().numpy(),
            's1_long': learner._repr_fn(batch.s1_long).cpu().numpy(),
            's2_long': learner._repr_fn(batch.s2_long).cpu().numpy(),
            's_neg': learner._repr_fn(batch.s_neg).cpu().numpy()
        }
    
    save_file = os.path.join(save_path, 'dual_discount_representations.npz')
    np.savez(save_file, **reprs)
    print(f"Saved dual-discount representations to {save_file}")


def main():
    timer = timer_tools.Timer()
    if FLAGS.log_sub_dir == 'auto_d':
        FLAGS.log_sub_dir = logging_tools.get_datetime()
    # pass args to config
    cfg_cls = get_config_cls()
    flags = flag_tools.Flags()
    flags.log_dir = os.path.join(
            FLAGS.log_base_dir,
            FLAGS.exp_name,
            FLAGS.env_id,
            FLAGS.log_sub_dir)
    flags.env_id = FLAGS.env_id
    flags.args = FLAGS.args
    logging_tools.config_logging(flags.log_dir)
    cfg = cfg_cls(flags)
    flag_tools.save_flags(cfg.flags, flags.log_dir)
    learner = laprepr.LapReprLearner(**cfg.args)
    learner.train()
        
    # save dual-discount representations after training
    save_dual_discount_representations(learner, flags.log_dir)
        
    print('Total time cost: {:.4g}s.'.format(timer.time_cost()))


if __name__ == '__main__':
    main()
