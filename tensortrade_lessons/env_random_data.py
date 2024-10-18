import ray
import numpy as np

from ray import tune
from ray.tune.registry import register_env

import tensortrade.env.default as default

from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio


USD = Instrument("USD", 2, "U.S. Dollar")
TTC = Instrument("TTC", 8, "TensorTrade Coin")

config = {"window_size":5}
          

x = np.arange(0, 2*np.pi, 2*np.pi / 1000)
p = Stream.source(50*np.sin(3*x) + 100, dtype="float").rename("USD-TTC")

bitfinex = Exchange("bitfinex", service=execute_order)(
    p
)

cash = Wallet(bitfinex, 100000 * USD)
asset = Wallet(bitfinex, 0 * TTC)

portfolio = Portfolio(USD, [
    cash,
    asset
])

feed = DataFeed([
    p,
    p.rolling(window=10).mean().rename("fast"),
    p.rolling(window=50).mean().rename("medium"),
    p.rolling(window=100).mean().rename("slow"),
    p.log().diff().fillna(0).rename("lr")
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


print("done")

