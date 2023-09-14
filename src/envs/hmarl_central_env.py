import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from envs.config import class_types, slice_req_params, total_res
from utils.utility import Utility


class CentralAgentEnv(gym.Env):
    def __init__(self):
        self.slice_class = "C1"
        self.utility = Utility()

        self.tcpu = total_res["C"]
        self.tbw = total_res["B"]
        self.tm = total_res["M"]

        self.av_cpu = self.tcpu
        self.av_bw = self.tbw
        self.av_mem = self.tm

        self.req_cpu = 0
        self.req_bw = 0
        self.req_mem = 0
        self.sl_duration = 0
        self.sl_bidvalue = 0
        self.val_generator()
        self.class_dict = {"C1": 0, "C2": 1, "C3": 2}
        self.observation_space = spaces.Dict(
            {
                "av_cpu": spaces.Box(
                    low=0,
                    high=self.tcpu + 1,
                    shape=(1,),
                    dtype=float,
                ),
                "av_mem": spaces.Box(
                    low=0,
                    high=self.tm + 1,
                    shape=(1,),
                    dtype=float,
                ),
                "av_bw": spaces.Box(
                    low=0,
                    high=self.tbw + 1,
                    shape=(1,),
                    dtype=float,
                ),
                "req_cpu": spaces.Box(
                    low=0,
                    high=2 * self.mean_req_cpu,
                    shape=(1,),
                    dtype=float,
                ),
                "req_bw": spaces.Box(
                    low=0,
                    high=2 * self.mean_req_bw,
                    shape=(1,),
                    dtype=float,
                ),
                "req_mem": spaces.Box(
                    low=0,
                    high=2 * self.mean_req_mem,
                    shape=(1,),
                    dtype=float,
                ),
                "duration": spaces.Box(
                    low=0,
                    high=2 * self.mean_duration,
                    shape=(1,),
                    dtype=float,
                ),
                "bid_value": spaces.Box(
                    low=0,
                    high=2 * self.mean_bid_value,
                    shape=(1,),
                    dtype=float,
                ),
                "slice_class": spaces.Discrete(3),
            }
        )

        self.action_space = spaces.Discrete(2)

        return

    def val_generator(self):
        self.mean_req_cpu = slice_req_params[self.slice_class]["NC"]
        self.mean_req_bw = slice_req_params[self.slice_class]["NB"]
        self.mean_req_mem = slice_req_params[self.slice_class]["NM"]
        self.mean_arr_rate = slice_req_params[self.slice_class]["arr_rate_mean"]
        self.mean_bid_value = slice_req_params[self.slice_class]["mean_bid_val"]
        self.mean_duration = slice_req_params[self.slice_class]["mean_duration"]
        self.mean_acc_slav = slice_req_params[self.slice_class]["SLAV"]
        self.mean_slav_penalty = slice_req_params[self.slice_class]["slav_penalty"]
        self.mean_rej_penalty = slice_req_params[self.slice_class]["rej_penalty"]
        self.var = slice_req_params[self.slice_class]["var"]

    def _get_obs(self):
        observation = {
            "av_cpu": self.av_cpu,
            "av_mem": self.av_mem,
            "av_bw": self.av_bw,
            "req_cpu": self.req_cpu,
            "req_bw": self.req_bw,
            "req_mem": self.req_mem,
            "duration": self.sl_duration,
            "bid_value": self.sl_bidvalue,
            "slice_class": self.class_dict[self.slice_class],
        }
        return observation

    def _get_info(self):
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.slice_class = np.random.choice(["C1", "C2", "C3"], size=1)[0]

        self.tcpu = np.array([int(total_res["C"])])
        self.tbw = np.array([int(total_res["B"])])
        self.tm = np.array([int(total_res["M"])])
        self.av_cpu = self.tcpu
        self.av_bw = self.tbw
        self.av_mem = self.tm

        self.val_generator()
        self.req_cpu = self.utility.get_value(self.mean_req_cpu, self.var)
        self.req_bw = self.utility.get_value(self.mean_req_bw, self.var)
        self.req_mem = self.utility.get_value(self.mean_req_mem, self.var)
        self.sl_duration = self.utility.get_value(self.mean_duration, self.var)
        self.sl_bidvalue = self.utility.get_value(self.mean_bid_value, self.var)

        observation = self._get_obs()
        info = {"reset_status": np.array([self.av_cpu, self.av_bw, self.av_mem])}

        return observation, info

    def step(self, action):
        self.val_generator()
        terminated = False
        reward = 0
        if (
            self.av_cpu >= self.req_cpu
            and self.av_mem >= self.req_mem
            and self.av_bw >= self.req_bw
        ):
            if action == 1:
                self.av_cpu = self.av_cpu - self.req_cpu
                self.av_mem = self.av_mem - self.req_mem
                self.av_bw = self.av_bw - self.req_bw
                reward = self.sl_duration * self.sl_bidvalue

                if self.utility.generate_slav(self.slice_class):
                    reward = reward - np.array([self.mean_slav_penalty])

                reward = reward[0]

            else:
                reward = int(self.mean_rej_penalty)

            self.slice_class = np.random.choice(["C1", "C2", "C3"], size=1)[0]
            self.req_cpu = self.utility.get_value(self.mean_req_cpu, self.var)
            self.req_bw = self.utility.get_value(self.mean_req_bw, self.var)
            self.req_mem = self.utility.get_value(self.mean_req_mem, self.var)
            self.sl_duration = self.utility.get_value(self.mean_duration, self.var)
            self.sl_bidvalue = self.utility.get_value(self.mean_bid_value, self.var)

        else:
            terminated = True

        observation = self._get_obs()
        info = {"step_status": np.array([self.av_cpu, self.av_bw, self.av_mem])}
        return observation, reward, terminated, False, info
