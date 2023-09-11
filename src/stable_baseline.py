import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v2", render_mode="rgb_array")
model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10000)

episodes = 10000

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        obs, reward, done, info, _ = env.step(env.action_space.sample())
