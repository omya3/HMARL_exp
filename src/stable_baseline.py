import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO

from envs.hmarl_central_env import CentralAgentEnv
from envs.hmarl_slice_env import ElasticSliceAgentEnv

slice_class = "C3"
algo = "PPO"
model_type = "agent"

if model_type == "central":
    env = CentralAgentEnv()
    models_dir = f"hmarl2/models/{model_type}_{algo}"
    log_file_name = f"central_{algo}"
else:
    env = ElasticSliceAgentEnv(slice_class=slice_class)
    models_dir = f"hmarl2/models/agent_{algo}_{slice_class}"
    log_file_name = f"{algo}_{slice_class}"

log_dir = "hmarl2/logs"

if algo == "PPO":
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)
elif algo == "A2C":
    model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)
else:
    model = DQN("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 100000

env.reset()
for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=log_file_name)
    model.save(f"{models_dir}/{TIMESTEPS*i}")
env.close()
