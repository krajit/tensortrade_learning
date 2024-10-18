import yfinance as yf
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.env.default import create
from tensortrade.env.default.actions import ManagedRiskOrders
from tensortrade.env.default.rewards import SimpleProfit
from tensortrade.oms.wallets import Wallet
from tensortrade.oms.wallets import Portfolio
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.exchanges import Exchange

# 1. Define the Instruments
INR = Instrument("INR", 2)  # Base currency
RELIANCE = Instrument("RELIANCE", 2)  # Stock

# 2. Fetch Historical Data for Reliance
data = yf.download('RELIANCE.NS', start='2022-01-01', end='2024-01-01')

# 3. Create Data Streams
price_stream = Stream.source(data['Close'].tolist(), dtype="float").rename("RELIANCE_CLOSE")
volume_stream = Stream.source(data['Volume'].tolist(), dtype="float").rename("RELIANCE_VOLUME")

# 4. Set up DataFeed
feed = DataFeed([price_stream, volume_stream])
feed.compile()

# 5. Define Wallets and Portfolio
exchange = Exchange("yfinance", service=feed) #, dtype="float")  # Exchange interface
wallet_inr = Wallet(exchange, 100000 * INR)  # Starting with 100,000 INR
wallet_stock = Wallet(exchange, 0 * RELIANCE)  # No stock initially

portfolio = Portfolio(INR, [wallet_inr, wallet_stock])  # Portfolio setup

# 6. Create the TensorTrade Environment
env = create(
    feed=feed,
    portfolio=portfolio,
    action_scheme=ManagedRiskOrders(),
    reward_scheme=SimpleProfit(),
    window_size=20
)

# 7. Test the Environment (Optional)
obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Sample a random action
    obs, reward, done, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}")