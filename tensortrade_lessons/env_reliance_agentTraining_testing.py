import yfinance as yf
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.env.default import create
from tensortrade.env.default.actions import ManagedRiskOrders
from tensortrade.env.default.rewards import SimpleProfit
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
import tensortrade.env.default as default
import pandas as pd, numpy as np


# 1. Define Instruments
INR = Instrument("INR", 2, "Indian Rupees")         # Base currency with 2 decimal places
RELIANCE = Instrument("RELIANCE", 8, 'Reliance Ind Ltd')  # Stock with 2 decimal places

# 2. Fetch Historical Data for Reliance
#data = yf.download('RELIANCE.NS', start='2022-01-01', end='2023-01-01').dropna()
#data.to_csv('datafile.csv')



data = pd.read_csv('datafile.csv')

# 3. Create Data Streams
price_stream = Stream.source(np.array(data["Close"]).reshape(-1), dtype="float").rename("INR-RELIANCE")
volume_stream = Stream.source(np.array(data["Volume"]).reshape(-1), dtype="float").rename("RELIANCE_VOLUME")


# 4. Set up DataFeed
feed = DataFeed([price_stream, volume_stream ])
#feed.compile()

# 5. Define Wallets and Portfolio
exchange = Exchange("yfinance", service=execute_order)( price_stream )

cash = Wallet(exchange, 100000 * INR)  # 100,000 INR
asset = Wallet(exchange, 0 * RELIANCE)  # Initially no stock

portfolio = Portfolio(INR, [cash, asset])

#reward_scheme = default.rewards.PBR(price=price_stream)
reward_scheme = default.rewards.SimpleProfit()


# action_scheme = default.actions.BSH(
#     cash=cash,
#     asset=asset
# ).attach(reward_scheme)
action_scheme = default.actions.SimpleOrders()


# 6. Create the TensorTrade Environment
env = create(
    feed=feed,
    portfolio=portfolio,
    action_scheme=action_scheme,
    reward_scheme=reward_scheme,
    window_size=20,
    max_allowed_loss=0.6
)

# 7. Test the Environment
obs = env.reset()
done = False
truncated = False

# while not done:
#     action = env.action_space.sample()  # Take a random action
#     obs, reward, done, truncated, info = env.step(action)
#     print(f"Action: {action}, Reward: {reward}")



# portfolio.ledger.as_frame().to_csv("ledger.csv")
# print("done")

import ray
import numpy as np
import pandas as pd

from ray import tune
from ray.tune.registry import register_env

from gymnasium.wrappers import FlattenObservation
def create_env(config):
    
    return FlattenObservation(env)

register_env("TradingEnv", create_env)


from ray.rllib.algorithms.ppo import PPOConfig

config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment("TradingEnv")
    .env_runners(num_env_runners=2)
    .framework("torch")
    .training(model={"fcnet_hiddens": [64, 64]})
    .evaluation(evaluation_num_env_runners=1)
)

algo = config.build()  # 2. build the algorithm,

# for _ in range(5):
#     print(algo.train())  # 3. train it,g

# algo.evaluate()  # 4. and evaluate it.

# Save the trained model to a directory
checkpoint_dir = "/tmp/tmpusp4uxgp"
# algo.save(checkpoint_dir)
# save_result = algo.save()
# #print(f"Model saved at {checkpoint_dir}")
# path_to_checkpoint = save_result.checkpoint.path
# print(
#     "An Algorithm checkpoint has been created inside directory: "
#     f"'{path_to_checkpoint}'."
# )

#import os
#checkpoint_dir = os.path.join("C:\\Users\\ajit.kumar\\Documents\\GitHub\\tensortrade_learning\\ppo_trading_model_reliance")

algo.restore(checkpoint_dir)
print("restoration done")



import pandas as pd
import matplotlib.pyplot as plt

import pathlib
import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.rl_module import RLModule

env = gym.make("CartPole-v1")

# Create only the neural network (RLModule) from our checkpoint.
rl_module = RLModule.from_checkpoint(
    pathlib.Path(checkpoint_dir.path) / "learner_group" / "learner" / "rl_module"
)["default_policy"]

episode_return = 0
terminated = truncated = False

obs, info = env.reset()

while not terminated and not truncated:
    # Compute the next action from a batch (B=1) of observations.
    torch_obs_batch = torch.from_numpy(np.array([obs]))
    action_logits = rl_module.forward_inference({"obs": torch_obs_batch})[
        "action_dist_inputs"
    ]
    # The default RLModule used here produces action logits (from which
    # we'll have to sample an action or use the max-likelihood one).
    action = torch.argmax(action_logits[0]).numpy()
    obs, reward, terminated, truncated, info = env.step(action)
    episode_return += reward

print(f"Reached episode return of {episode_return}.")

performance = pd.DataFrame.from_dict(env.action_scheme.portfolio.performance, orient='index')
performance.plot()
plt.show()
print("done")