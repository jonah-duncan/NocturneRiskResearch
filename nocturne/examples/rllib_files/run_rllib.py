# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Example run script for RLlib."""
import os

import hydra
from omegaconf import OmegaConf
from cfgs.config import set_display_window
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from nocturne.envs.wrappers import create_env


class RLlibWrapperEnv(MultiAgentEnv):
    """Thin wrapper making our env look like a MultiAgentEnv."""

    metadata = {
        "render.modes": ["rgb_array"],
    }

    def __init__(self, env):
        """See wrapped env class."""
        self._skip_env_checking = True  # temporary fix for rllib env checking issue
        super().__init__()
        self._env = env

    def step(self, actions):
        """See wrapped env class."""
        next_obs, rew, done, info = self._env.step(actions)
        return next_obs, rew, done, info

    def reset(self):
        """See wrapped env class."""
        obses = self._env.reset()
        return obses

    @property
    def observation_space(self):
        """See wrapped env class."""
        return self._env.observation_space

    @property
    def action_space(self):
        """See wrapped env class."""
        return self._env.action_space

    def render(self, mode=None):
        """See wrapped env class."""
        return self._env.render()

    def seed(self, seed=None):
        """Set seed on the wrapped env."""
        self._env.seed(seed)
    
    #''' Original Code
    def __getattr__(self, name):
        """Return attributes from the wrapped env."""
        return getattr(self._env, name)
    #'''

    ''' Tried Solution
    @property
    def __getattribute__(self, name):
        print('getting: ' + name)
        try:
            return super(RLlibWrapperEnv, self).__getattribute__(name)
        except AttributeError:
            print('oh no, AttributeError caught and reraising')
            raise

    def __getattr__(self, name):
        """Called if __getattribute__ raises AttributeError"""
        return 'close but no ' + name
    '''

    ''' Tried Solution
    @property
    def __getattr__(self, key):
        if key in self.__dict__['self._env']:
            return self.__dict__['self._env'][key]
    '''

    ''' Yet another Tried Solution
    def __getattr__(self, name):
        if name == "special":
            raise AttributeError()
        if name in self.special:
            return "yes"
        raise AttributeError()
    '''
    
    ''' Another Tried Solution
    def __getattribute__(self, item):
        print('__getattribute__ ', item)
        # Calling the super class to avoid recursion
        return super(RLlibWrapperEnv, self).__getattribute__(item)    # Gets called when the item is not found via __getattribute__
    def __getattr__(self, item):
        print('__getattr__ ', item)
        return super(RLlibWrapperEnv, self).__setattr__(item, 'orphan')
    '''

def create_rllib_env(cfg):
    """Return an MultiAgentEnv wrapped environment."""
    return RLlibWrapperEnv(create_env(cfg))


@hydra.main(config_path="/cluster/research-groups/ramasubramanian/workspace/duncanj5/nocturne/cfgs/", config_name="config")
def main(cfg):
    """Run RLlib example."""
    set_display_window()
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # TODO(eugenevinitsky) move these into a config
    if cfg['debug']:
        ray.init(local_mode=True)
        num_workers = 0
        num_envs_per_worker = 1
        num_gpus = 0
        use_lstm = False
    else:
        num_workers = 15
        num_envs_per_worker = 5
        num_gpus = 1
        use_lstm = True

    register_env("nocturne", lambda cfg: create_rllib_env(cfg))

    username = os.environ["USER"]

    ray.init()
    #ray.init()
    tune.run(
        "DQN", # This seems to change what agent is used
        # TODO(eugenevinitsky) move into config
        local_dir=f"/cluster/research-groups/ramasubramanian/workspace/{username}/nocturne/ray_results",
        stop={"episodes_total": 60000},
        checkpoint_freq=1000,
        config={
            # Enviroment specific.
            "env":
            "nocturne",
            "env_config":
            cfg,
            # General
            "framework":
            "torch",
            "num_gpus":
            num_gpus,
            "num_workers":
            num_workers,
            "num_envs_per_worker":
            num_envs_per_worker,
            "observation_filter":
            "MeanStdFilter",
            # Evaluation stuff
            "evaluation_interval":
            50,
            # Run evaluation on (at least) one episodes
            "evaluation_duration":
            1,
            # ... using one evaluation worker (setting this to 0 will cause
            # evaluation to run on the local evaluation worker, blocking
            # training until evaluation is done).
            # TODO: if this is not 0, it seems to error out
            "evaluation_num_workers":
            0,
            # Special evaluation config. Keys specified here will override
            # the same keys in the main config, but only for evaluation.
            "evaluation_config": {
                # Store videos in this relative directory here inside
                # the default output dir (~/ray_results/...).
                # Alternatively, you can specify an absolute path.
                # Set to True for using the default output dir (~/ray_results/...).
                # Set to False for not recording anything.
                "record_env": "videos_test",
                # "record_env": "/Users/xyz/my_videos/",
                # Render the env while evaluating.
                # Note that this will always only render the 1st RolloutWorker's
                # env and only the 1st sub-env in a vectorized env.
                "render_env": True,
            },
        },
    )


if __name__ == "__main__":
    main()


# Method specific.
'''
"entropy_coeff":
0.0,
"num_sgd_iter":
5,
"train_batch_size":
max(100 * num_workers * num_envs_per_worker, 512),
"rollout_fragment_length":
20,
"sgd_minibatch_size":
max(int(100 * num_workers * num_envs_per_worker / 4), 512),
"multiagent": {
# We only have one policy (calling it "shared").
# Class, obs/act-spaces, and config will be derived
# automatically.
"policies": {"shared_policy"},
# Always use "shared" policy.
"policy_mapping_fn":
(lambda agent_id, episode, **kwargs: "shared_policy"),
# each agent step is counted towards train_batch_size
# rather than environment steps
"count_steps_by":
"agent_steps",
},
"model": {
"use_lstm": use_lstm
},
'''


'''
tune.run(
        "DQN", # This seems to change what agent is used
        # TODO(eugenevinitsky) move into config
        local_dir=f"/cluster/research-groups/ramasubramanian/workspace/{username}/nocturne/ray_results",
        stop={"episodes_total": 60000},
        checkpoint_freq=1000,
        config={
            # Enviroment specific.
            "env":
            "nocturne",
            "env_config":
            cfg,
            # General
            "framework":
            "torch",
            "num_gpus":
            num_gpus,
            "num_workers":
            num_workers,
            "num_envs_per_worker":
            num_envs_per_worker,
            "observation_filter":
            "MeanStdFilter",
            "train_batch_size":
            max(100 * num_workers * num_envs_per_worker, 512),
            "rollout_fragment_length":
            20,
            # Evaluation stuff
            "evaluation_interval":
            50,
            # Run evaluation on (at least) one episodes
            "evaluation_duration":
            1,
            # ... using one evaluation worker (setting this to 0 will cause
            # evaluation to run on the local evaluation worker, blocking
            # training until evaluation is done).
            # TODO: if this is not 0, it seems to error out
            "evaluation_num_workers":
            0,
            # Special evaluation config. Keys specified here will override
            # the same keys in the main config, but only for evaluation.
            "evaluation_config": {
                # Store videos in this relative directory here inside
                # the default output dir (~/ray_results/...).
                # Alternatively, you can specify an absolute path.
                # Set to True for using the default output dir (~/ray_results/...).
                # Set to False for not recording anything.
                "record_env": "videos_test",
                # "record_env": "/Users/xyz/my_videos/",
                # Render the env while evaluating.
                # Note that this will always only render the 1st RolloutWorker's
                # env and only the 1st sub-env in a vectorized env.
                "render_env": True,
            },
        },
    )
'''
