from stable_baselines3.common.env_checker import check_env

from hmarl_central_env import CentralAgentEnv
from hmarl_slice_env import ElasticSliceAgentEnv

env = CentralAgentEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)
