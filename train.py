from datetime import datetime
from stable_baselines3.ppo.ppo import PPO
from frankenstein.components.environment.trading.ml.environement import TradingEnv
from frankenstein.components.environment.trading.data_provider import DataProvider
from frankenstein.lib.trading.utils import load_mt5_bars_csv, load_mt5_ticks_csv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback
import torch as th
import numpy as np

th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False

th.manual_seed(0)
np.random.seed(0)

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

class TrainCallback(BaseCallback):
 
    def __init__(self, verbose=1):
        super(TrainCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.num_timesteps % 50000 == 0:
            self.training_env.env_method("reset_stats")
        if self.num_timesteps % 2048 == 0:
            self.logger.record("timesteps", self.num_timesteps)
            stats = self.training_env.env_method("get_stats")
            for key, value in stats[0].items():
                self.logger.record(key, value)
            self.logger.dump(self.num_timesteps)
        if self.num_timesteps % 10000 == 0:
            self.model.save("dqn_policy")
        return True


env_params = [
    'M5', # frequency
    20, # n_rolling_observations
]


data_provider, start, end = get_data_provider("datasets/EURUSD_SB_M1_202001020000_202405292358.csv")

env = TradingEnv(
    *([data_provider, datetime.strftime(start, "%Y-%m-%dT%H:%M:%S.0"), datetime.strftime(end, "%Y-%m-%dT%H:%M:%S.0")] + env_params)
)

eval_data_provider, eval_start, eval_end = get_data_provider("datasets/EURUSD_SB_M1_201901020000_201905300000.csv")
eval_env = TradingEnv(
    *([eval_data_provider, datetime.strftime(eval_start, "%Y-%m-%dT%H:%M:%S.0"), datetime.strftime(eval_end, "%Y-%m-%dT%H:%M:%S.0")] + env_params)
)

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[1024, 1024, 1024], vf=[1024, 1024, 1024]))

model = PPO(
    "MlpPolicy", 
    env, 
    learning_rate=1e-4,
    batch_size=64,
    verbose=1,
    policy_kwargs=policy_kwargs
)

if __name__ == "__main__":
    new_logger = configure(None, ["stdout", "log", "csv"])
    model.set_logger(new_logger)

    model.learn(
        total_timesteps=5000000, 
        log_interval=100,
        callback=TrainCallback()
    )

    print(evaluate_policy(model, env, n_eval_episodes=1, return_episode_rewards=True))

