import numpy as np
import cv2
import gymnasium as gym
import ale_py
from copy import deepcopy
import torch
from torch import inf
import argparse
import pprint

# Register ALE
gym.register_envs(ale_py)

def mean_of_list(func):
    def function_wrapper(*args, **kwargs):
        lists = func(*args, **kwargs)
        return [sum(list) / len(list) for list in lists[:-4]] + [explained_variance(lists[-4], lists[-3])] + \
               [explained_variance(lists[-2], lists[-1])]
    return function_wrapper

def preprocessing(data):
    # Handle Gymnasium's (obs, info) tuple if passed directly
    if isinstance(data, tuple):
        img = data[0]
    else:
        img = data
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
    return img

def stack_states(stacked_frames, state, is_new_episode):
    frame = preprocessing(state)
    if is_new_episode:
        stacked_frames = np.stack([frame for _ in range(4)], axis=0)
    else:
        stacked_frames = stacked_frames[1:, ...]
        stacked_frames = np.concatenate([stacked_frames, np.expand_dims(frame, axis=0)], axis=0)
    return stacked_frames

def explained_variance(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary

def make_atari(env_id, max_episode_steps, sticky_action=True, max_and_skip=True):
    # frameskip=1 is essential for these wrappers to work as intended
    env = gym.make(env_id, render_mode="rgb_array", frameskip=1)
    
    if hasattr(env, '_max_episode_steps'):
        env._max_episode_steps = max_episode_steps * 4
        
    if sticky_action:
        env = StickyActionEnv(env)
    if max_and_skip:
        env = RepeatActionEnv(env)
        
    env = MontezumaVisitedRoomEnv(env, 3)
    env = AddRandomStateToInfoEnv(env)
    return env

# --- WRAPPERS UPDATED FOR GYMNASIUM (5 Values) ---

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
        total_reward = 0
        terminated = False
        truncated = False
        info = {}
        
        for t in range(4):
            # UNPACK 5 VALUES
            state, r, term, trunc, info = self.env.step(action)
            
            if t == 2:
                self.successive_frame[0] = state
            elif t == 3:
                self.successive_frame[1] = state
            
            total_reward += r
            
            if term or trunc:
                terminated = term
                truncated = trunc
                break

        state = self.successive_frame.max(axis=0)
        # RETURN 5 VALUES
        return state, total_reward, terminated, truncated, info

class MontezumaVisitedRoomEnv(gym.Wrapper):
    def __init__(self, env, room_address):
        gym.Wrapper.__init__(self, env)
        self.room_address = room_address
        self.visited_rooms = set()

    def step(self, action):
        # UNPACK 5 VALUES
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Try to access RAM via ale-py interface
        try:
            # In Gymnasium v1.0+, access might need unwrapping to the ALE instance
            if hasattr(self.unwrapped, 'ale'):
                ram = self.unwrapped.ale.getRAM()
                self.visited_rooms.add(ram[self.room_address])
        except Exception:
            pass # Fail silently if RAM access changes in future versions

        if terminated or truncated:
            if "episode" not in info:
                info["episode"] = {}
            info["episode"].update(visited_room=deepcopy(self.visited_rooms))
            self.visited_rooms.clear()
            
        # RETURN 5 VALUES
        return state, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class AddRandomStateToInfoEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.rng_at_episode_start = deepcopy(self.unwrapped.np_random)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            if 'episode' not in info:
                info['episode'] = {}
            info['episode']['rng_at_episode_start'] = self.rng_at_episode_start
        return state, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.rng_at_episode_start = deepcopy(self.unwrapped.np_random)
        return self.env.reset(**kwargs)

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count

def clip_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm

def get_params():
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument("--n_workers", default=2, type=int)
    parser.add_argument("--interval", default=50, type=int)
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--train_from_scratch", action="store_false")
    parser.add_argument('-f', '--file', help='Dummy argument for Jupyter')
    parser_params = parser.parse_args()

    default_params = {"env_name": "ALE/MontezumaRevenge-v5",
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
    total_params = {**vars(parser_params), **default_params}
    return total_params