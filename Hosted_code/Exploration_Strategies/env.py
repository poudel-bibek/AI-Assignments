import argparse
from copy import deepcopy

import ale_py
import gymnasium as gym
from gymnasium import error as gym_error
import numpy as np

_ALE_REGISTERED = False


def _ensure_ale_namespace():
    global _ALE_REGISTERED
    if _ALE_REGISTERED:
        return
    gym.register_envs(ale_py)
    _ALE_REGISTERED = True


_ensure_ale_namespace()

class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def step(self, action):
        if np.random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        return self.env.step(action)

    def reset(self, **kwargs):
        self.last_action = 0
        return self.env.reset(**kwargs)

class RepeatActionEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.successive_frame = np.zeros((2,) + self.env.observation_space.shape, dtype=np.uint8)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        reward, done = 0, False
        for t in range(4):
            state, r, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if t == 2:
                self.successive_frame[0] = state
            elif t == 3:
                self.successive_frame[1] = state
            reward += r
            if done:
                break
        state = self.successive_frame.max(axis=0)
        return state, reward, terminated, truncated, info

class MontezumaVisitedRoomEnv(gym.Wrapper):
    def __init__(self, env, room_address):
        gym.Wrapper.__init__(self, env)
        self.room_address = room_address
        self.visited_rooms = set()

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        ram = self.unwrapped.ale.getRAM()
        assert len(ram) == 128
        self.visited_rooms.add(ram[self.room_address])
        if done:
            if "episode" not in info:
                info["episode"] = {}
            info["episode"].update(visited_room=deepcopy(self.visited_rooms))
            self.visited_rooms.clear()
        return state, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class AddRandomStateToInfoEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.rng_at_episode_start = deepcopy(self.unwrapped.np_random)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if done:
            if 'episode' not in info:
                info['episode'] = {}
            info['episode']['rng_at_episode_start'] = self.rng_at_episode_start
        return state, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.rng_at_episode_start = deepcopy(self.unwrapped.np_random)
        return self.env.reset(**kwargs)

def make_atari(env_id, max_episode_steps, sticky_action=True, max_and_skip=True):
    """Create and wrap Atari environment"""
    _ensure_ale_namespace()
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except gym_error.NamespaceNotFound:
        # Retry once after forcing registration
        _ensure_ale_namespace()
        env = gym.make(env_id, render_mode="rgb_array")
    env._max_episode_steps = max_episode_steps * 4
    if sticky_action:
        env = StickyActionEnv(env)
    if max_and_skip:
        env = RepeatActionEnv(env)
    env = MontezumaVisitedRoomEnv(env, 3)
    env = AddRandomStateToInfoEnv(env)
    return env


def get_params():
    """Get configuration parameters (CLI-friendly; works in notebooks too)."""
    parser = argparse.ArgumentParser(description="RND Config")
    parser.add_argument("--n_workers", default=2, type=int)
    parser.add_argument("--interval", default=50, type=int)
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--train_from_scratch", action="store_false")
    parser.add_argument("-f", "--file", help="Dummy argument for Jupyter")

    parser_params = parser.parse_args()

    default_params = {
        "env_name": "ALE/MontezumaRevenge-v5",
        "state_shape": (4, 84, 84),
        "obs_shape": (1, 84, 84),
        "total_rollouts_per_env": int(30e3),
        "max_frames_per_episode": 4500,
        "rollout_length": 128,
        "n_epochs": 4,
        "n_mini_batch": 4,
        "lr": 1e-4,
        "ext_gamma": 0.999,
        "int_gamma": 0.99,
        "lambda": 0.95,
        "ext_adv_coeff": 2,
        "int_adv_coeff": 1,
        "ent_coeff": 0.001,
        "clip_range": 0.1,
        "pre_normalization_steps": 50,
    }

    total_params = {**default_params, **vars(parser_params)}
    return total_params
