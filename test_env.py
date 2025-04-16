import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

import frasa_env

gym.register_envs(frasa_env)

argparser = argparse.ArgumentParser(description="Test the frasa-standup-v0 environment")
argparser.add_argument("--env", type=str, default="frasa-standup-v0", help="Environment to test")
argparser.add_argument("--random", action="store_true", help="Use random actions instead of zeros")
argparser.add_argument("--normal", action="store_true", help="Use normal action noise")
argparser.add_argument("--orn", action="store_true", help="Use Ornstein-Uhlenbeck action noise")
argparser.add_argument("--std", type=float, default=0.1, help="Standard deviation for the action noise")
argparser.add_argument("--theta", type=float, default=0.15, help="Theta for the Ornstein-Uhlenbeck noise")
args = argparser.parse_args()

env = gym.make(args.env)
env.reset()
noise = None
returns = 0
step = 0

if args.normal:
    noise = NormalActionNoise(
        mean=np.zeros(env.action_space.shape[0]),
        sigma=args.std * np.ones(env.action_space.shape[0]),
    )
elif args.orn:
    noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(env.action_space.shape[0]),
        sigma=args.std * np.ones(env.action_space.shape[0]),
        theta=args.theta,
    )

while True:
    step += 1
    action = env.action_space.sample()

    if not args.random:
        action = np.zeros_like(action)

    if noise is not None:
        action += noise()

    obs, reward, done, trucated, infos = env.step(action)
    returns += reward
    env.render()

    if done or trucated or step > 30:
        status = "truncated" if trucated else "done"
        print(f"Episode finished ({status}) after {step} steps, returns: {returns}")
        step = 0
        returns = 0
        env.reset()
