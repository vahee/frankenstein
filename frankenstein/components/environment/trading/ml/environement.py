from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import gymnasium as gym
import numpy as np
from gym import spaces
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
        
        bands_timeframe: str = 'M10', 
        bands_window: int = 7, 
        bands_dev: int = 2, 
        
        rsi_timeframe: str = 'M10', 
        rsi_period: int = 13,
        
        stochastic_timeframe: str = 'M10',
        stochastic_smooth: int = 3,
        stochastic_period: int = 14
    ):
        super().__init__()
        
        self._data_provider = data_provider
        
        self._time_start = time_start
        self._time_end = time_end
        self._freq = freq
        
        self.symbol = 'EURUSD'
        
        self._position = 0
        self._entry_price = 0
        self._timestep: datetime
        self._last_n_observations = []
        self._n_rolling_observations = 6
        self._last_action = None
        
        self._equity = 10000
        self._prepare(
            
            bands_timeframe=bands_timeframe, 
            bands_window=bands_window, 
            bands_dev=bands_dev, 
            
            rsi_timeframe=rsi_timeframe, 
            rsi_period=rsi_period,
            
            stochastic_timeframe=stochastic_timeframe,
            stochastic_smooth=stochastic_smooth,
            stochastic_period=stochastic_period
        )
        
        self.action_space = spaces.Discrete(3) # Buy, Sell, Hold
        
        # (4 bars, 9 features - open, low, high, close, initial position, rsi, hband, lband, stochastic)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self._n_rolling_observations , 9), dtype=np.float16)
        
    def _prepare(self, 
        *,
        
        
        bands_timeframe: str, 
        bands_window: int, 
        bands_dev: int, 
        
        rsi_timeframe: str, 
        rsi_period: int,
        
        stochastic_timeframe: str,
        stochastic_period: int,
        stochastic_smooth: int,
    ) -> None:
        
        
        self._params = {
            
            'bands_timeframe': bands_timeframe,
            'bands_window': bands_window,
            'bands_dev': bands_dev,
            
            'rsi_timeframe': rsi_timeframe,
            'rsi_period': rsi_period,
            
            'stochastic_timeframe': stochastic_timeframe,
            'stochastic_period': stochastic_period,
            'stochastic_smooth': stochastic_smooth,
        }
        
        # bands
        
        self.bands_bars = self._data_provider.bars(self.symbol, bands_timeframe)
        
        self.bands_bars['hband'] = volatility.bollinger_hband(
            self.bands_bars['close'], window=int(bands_window), window_dev=int(bands_dev))
        self.bands_bars['lband'] = volatility.bollinger_lband(
            self.bands_bars['close'], window=int(bands_window), window_dev=int(bands_dev))
        
        self.bands_bars['mband'] = volatility.bollinger_mavg(
            self.bands_bars['close'], window=int(bands_window))
        
        # rsi
        
        self.rsi_bars = self._data_provider.bars(self.symbol, rsi_timeframe)
        
        self.rsi_bars['rsi'] = momentum.rsi(self.rsi_bars['close'], window=int(rsi_period))
        
        # stochastic
        
        self.stochastic_bars = self._data_provider.bars(self.symbol, stochastic_timeframe)
        
        self.stochastic_bars['stoch'] = momentum.stoch(self.stochastic_bars['high'], self.stochastic_bars['low'], self.stochastic_bars['close'], window=int(stochastic_period), smooth_window=int(stochastic_smooth))
                
    def _observe(self):
        if self._timestep is None:
            return
        rsi = self.rsi_bars.loc[:self._timestep + timedelta(microseconds=1)].iloc[-1]['rsi']
        hband = self.bands_bars.loc[:self._timestep + timedelta(microseconds=1)].iloc[-1]['hband']
        lband = self.bands_bars.loc[:self._timestep + timedelta(microseconds=1)].iloc[-1]['lband']
        high = self.stochastic_bars.loc[:self._timestep + timedelta(microseconds=1)].iloc[-1]['high']
        low = self.stochastic_bars.loc[:self._timestep + timedelta(microseconds=1)].iloc[-1]['low']
        close = self.stochastic_bars.loc[:self._timestep + timedelta(microseconds=1)].iloc[-1]['close']
        open = self.stochastic_bars.loc[:self._timestep + timedelta(microseconds=1)].iloc[-1]['open']
        stochastic = self.stochastic_bars.loc[:self._timestep + timedelta(microseconds=1)].iloc[-1]['stoch']
        
        self._last_n_observations.append(
            [
                open,
                low,
                high,
                close, 
                self._position, 
                rsi, 
                hband, 
                lband, 
                stochastic
            ])
        
        if len(self._last_n_observations) > self._n_rolling_observations:
            self._last_n_observations.pop(0)
    
    def _reward(self) -> float:
        if self._timestep is None:
            return 0
        
        new_position = self._last_n_observations[-1][4]
        prev_price = self._last_n_observations[-2][3]
        new_price = self._last_n_observations[-1][3]
        
        reward = 0
        if new_position == 1:
            reward = new_price - prev_price
            self._equity += reward
        elif new_position == -1:
            reward = prev_price - new_price
            self._equity += reward
        
        return reward
        
    def step(self, action):
        
        # action: 0 - Buy, 1 - Sell, 2 - Hold
        current_position = self._last_n_observations[-1][4]
        if action == 0:
            if current_position == 0:
                self._position = 1
            elif current_position == -1:
                self._position = 0
            
        elif action == 1:
            if current_position == 0:
                self._position = -1
            elif current_position == 1:
                self._position = 0
        else:
            self._position = current_position
        
        self._last_action = action
        self._data_provider.step()
        self._timestep = self._data_provider.get_time()
        
        done = self._timestep is None
        
        self._observe()
        reward = self._reward()
        
        observation = np.array(self._last_n_observations, dtype=np.float16)
        
        return observation, reward, done, {}

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
        return observation

    def render(self, mode='human'):
        print(f'Equity: {self._equity}, Position: {self._position}, Last Action: {self._last_action}')
    
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