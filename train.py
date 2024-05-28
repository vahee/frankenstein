from datetime import datetime
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.monitor import Monitor
from frankenstein.components.environment.trading.ml.environement import TradingEnv
from frankenstein.components.environment.trading.data_provider import DataProvider
from frankenstein.lib.trading.utils import load_mt5_bars_csv, load_mt5_ticks_csv

def get_data_provider(filename):
    bars = True

    if bars:
        df = load_mt5_bars_csv(filename)
    else:
        df = load_mt5_ticks_csv(filename)

    symbol = "EURUSD"

    assert symbol is not None, "Symbol is not set"

    try:
        start, end = df['timestamp'].iloc[0], df['timestamp'].iloc[-1]
    except:
        start, end = df['timestamp'][0], df['timestamp'][-1]

    start = start.replace(microsecond=0)
    end = end.replace(microsecond=0)

    data_provider = DataProvider()
    data_provider.load_ticks_pd_dataframe(df, symbol)
    return data_provider, start, end

data_provider, start, end = get_data_provider("datasets/NAS100_SB_M1_202401020100_202405172354.csv")

env = TradingEnv(
    data_provider, 
    datetime.strftime(start, "%Y-%m-%dT%H:%M:%S.0"),
    datetime.strftime(end, "%Y-%m-%dT%H:%M:%S.0"),
    'M5', 
    'M5', 
    13, 
    2, 
    'M5', 
    9, 
    'M5', 
    3, 
    14
)


eval_data_provider, eval_start, eval_end = get_data_provider("datasets/NAS100_SB_M1_202001020100_202005282246.csv")
eval_env = TradingEnv(
    eval_data_provider, 
    datetime.strftime(eval_start, "%Y-%m-%dT%H:%M:%S.0"),
    datetime.strftime(eval_end, "%Y-%m-%dT%H:%M:%S.0"),
    'M5', 
    'M5', 
    13, 
    2, 
    'M5', 
    9, 
    'M5', 
    3, 
    14
)
m_eval_env =  Monitor(env)

# Define and Train the agent
m_env = Monitor(env)
model = DQN("MlpPolicy", m_env, learning_rate=1e-4, verbose=1, target_update_interval=1000, create_eval_env=True, gradient_steps=100)
model.learn(
    total_timesteps=100000, 
    eval_env=m_eval_env,
    log_interval=100,
    n_eval_episodes=1
)
model.save("dqn_policy")

