from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from frankenstein.lib.trading.protocols import IDataProvider
from ta import volatility, momentum

class TradingEnv(gym.Env):
    """Trading Environment that follows gym interface."""

    metadata = {
        "render.modes": ["human"],
    }

    def __init__(
        self, 
        data_provider: IDataProvider,
        
        time_start: str,
        time_end: str,
        freq: str,
        n_rolling_observations: int = 6

    ):
        super().__init__()
        
        self._data_provider = data_provider
        
        self._time_start = time_start
        self._time_end = time_end
        self._freq = freq
        
        self.symbol = 'EURUSD'
        self._lot_multiplier = 10000
        self._position = 0
        self._entry_price = 0
        self._timestep: datetime
        self._last_n_observations = []
        self._last_n_observations_abs = []
        self._n_rolling_observations = n_rolling_observations
        self._last_action = None
        
        self._equity = 10000
        
        self.action_space = spaces.Discrete(4) # Buy, Sell, Hold, Close
        # (self._n_rolling_observations bars, 3 features - price, position, position price)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self._n_rolling_observations , 3), dtype=np.float16)            
    def _observe(self):
        if self._timestep is None:
            return
        price = self._data_provider.bid(self.symbol) * self._lot_multiplier
        self._last_n_observations_abs.append(
            [
                price, 
                self._position, 
                self._entry_price
            ])
        
        if len(self._last_n_observations_abs) > 1:
            self._last_n_observations.append(
                [
                    self._last_n_observations_abs[-1][0] - self._last_n_observations_abs[-2][0], 
                    self._position, 
                    self._entry_price,
                ])
        self._last_n_observations_abs = self._last_n_observations_abs[-self._n_rolling_observations:]
        self._last_n_observations = self._last_n_observations[-self._n_rolling_observations:]
    
    def _reward(self, action, old_observation, new_observation) -> float:
        if self._timestep is None:
            return 0
        
        new_position = new_observation[1]
        old_position = old_observation[1]
        
        reward = -5
        if action == 2:
            if old_position in [1, -1]:
                reward = old_position * (new_observation[2] - old_observation[2])
                self._equity += reward
                if reward > 0:
                    reward = 20 + 5 * reward
                else:
                    reward = -10 + 5 * reward
            else:
                reward = -10
        elif action in [0, 1]:
            if old_position != 0:
                reward = -10
            else:
                reward = 5
        return reward
        
    def step(self, action):
        
        # action: 0 - Buy, 1 - Sell, 2 - Close, 3 - Hold
        current_position = self._last_n_observations[-1][1]
        if action == 0: # buy
            if current_position == 0:
                self._position = 1
                self._entry_price = self._last_n_observations[-1][0]
            
        elif action == 1: # sell
            if current_position == 0:
                self._position = -1
                self._entry_price = self._last_n_observations[-1][0]
        elif action == 2: # close
            self._position = 0
            self._entry_price = 0
        else: # hold
            self._position = current_position
            self._entry_price = self._entry_price
        
        self._last_action = action
        self._data_provider.step()
        self._timestep = self._data_provider.get_time()
        
        done = self._timestep is None
        old_observation = self._last_n_observations[-1]
        self._observe()
        new_observation = self._last_n_observations[-1]
        reward = self._reward(action, old_observation, new_observation)
        
        observation = np.array(self._last_n_observations, dtype=np.float16)
        
        return observation, reward, done, done, {}

    def reset(self, seed=None, options=None):
        self._data_provider.reset(self._time_start, self._time_end, self._freq)
        self._equity = 10000
        self._position = 0
        self._last_n_observations = []
        
        while True:
            self._data_provider.step()
            self._timestep = self._data_provider.get_time()
            self._observe()
            observation = np.array(self._last_n_observations, dtype=np.float16)
            if np.all(~np.isnan(observation)) and observation.shape[0] == self._n_rolling_observations:
                break
            if self._timestep is None:
                break
        return observation, {}

    def render(self, mode='human'):
        print(self.get_stats())
    
    def get_stats(self):
        return {
            'equity': self._equity,
            'position': self._position,
            'last_action': self._last_action
        }
        
    def reset_stats(self):
        self._equity = 10000
        self._position = 0
    
    def close(self):
        ...