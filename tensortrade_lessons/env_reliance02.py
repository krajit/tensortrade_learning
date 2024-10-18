import ray
import numpy as np
import yfinance as yf
from ray import tune
from ray.tune.registry import register_env

import tensortrade.env.default as default

from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio


USD = Instrument("INR", 2, "Indian Rupees")
#TTC = Instrument("TTC", 8, "TensorTrade Coin")

config = {"window_size":30}
          

#x = np.arange(0, 2*np.pi, 2*np.pi / 1000)
#p = Stream.source(50*np.sin(3*x) + 100, dtype="float").rename("USD-TTC")

###
# 1. Define Instruments
INR = Instrument("INR", 2, "Indian Rupees")         # Base currency with 2 decimal places
RELIANCE = Instrument("RELIANCE", 8, 'Reliance Ind Ltd')  # Stock with 2 decimal places

# 2. Fetch Historical Data for Reliance
data = yf.download('RELIANCE.NS', start='2022-01-01', end='2023-01-01').dropna()

# 3. Create Data Streams
price_stream = Stream.source(data['Close'].tolist(), dtype="float").rename("RELIANCE")
#volume_stream = Stream.source(data['Volume'].tolist(), dtype="float").rename("RELIANCE_VOLUME")
###

p = Stream.source(data['Close'].tolist(), dtype="float").rename("INR-RELIANCE")



bitfinex = Exchange("bitfinex", service=execute_order)(
    p
)

cash = Wallet(bitfinex, 100000 * INR)
asset = Wallet(bitfinex, 0 * RELIANCE)

portfolio = Portfolio(INR, [
    cash,
    asset
])

feed = DataFeed([
    p,
    # p.rolling(window=10).mean().rename("fast"),
    # p.rolling(window=50).mean().rename("medium"),
    # p.rolling(window=100).mean().rename("slow"),
    # p.log().diff().fillna(0).rename("lr")
])

reward_scheme = default.rewards.PBR(price=p)

action_scheme = default.actions.BSH(
    cash=cash,
    asset=asset
).attach(reward_scheme)

env = default.create(
    feed=feed,
    portfolio=portfolio,
    action_scheme=action_scheme,
    reward_scheme=reward_scheme,
    window_size=config["window_size"],
    max_allowed_loss=0.6
)

env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Take a random action
    obs, reward, done, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}")


print("done")

