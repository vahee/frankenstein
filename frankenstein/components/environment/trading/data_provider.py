from datetime import datetime, timedelta, UTC
from typing import Tuple, Optional, Literal
from bisect import bisect_left
import pandas as pd


from agentopy import IEnvironmentComponent, WithActionSpaceMixin, IState, State, EntityInfo, Action, ActionResult
from frankenstein.lib.trading.protocols import IDataProvider
from frankenstein.lib.trading.utils import aggregate_prices

import logging

logger = logging.getLogger(__name__)


class DataProvider(WithActionSpaceMixin, IDataProvider, IEnvironmentComponent):
    def __init__(self, *, time_range: Optional[Tuple[datetime, datetime]] = None) -> None:
        super().__init__()
        
        self._time: datetime = None
        self._live: bool = False
        
        if time_range is not None:
            start, end = time_range
            
            assert isinstance(start, datetime), "time_range[0] must be datetime"
            assert isinstance(end, datetime), "time_range[1] must be datetime"
            assert start < end, "time_range[0] must be less than time_range[1]"
            
            self._min_start_time, self._min_end_time = start, end
            self._time = start
        
        self._cache = {}
        self._data = {}
        self._time_freq_str = None
        self._start_time, self._end_time, self._time_freq = None, None, None
        
        self.action_space.register_actions(
            [
                Action('data_provider_live', "start backtesting", self.live, self.info()),
                Action('data_provider_replay', "Loads ticks data from dataframe", self.replay, self.info()),
                Action('data_provider_stop', "stop backtesting", self.stop, self.info()),
            ]
        )
        
    def load_ticks_pd_dataframe(self, df: pd.DataFrame, symbol: str):
        self._data[symbol] = {
            'index': df.index.tolist(),
            'df': df,
            'dict': df.to_dict('index'),
            'source': 'pd.dataframe'
        }
    
    def load_ticks_dt_dataframe(self, df: pd.DataFrame, symbol: str):
        self._data[symbol] = {
            'df': df,
            'source': 'dt.dataframe'
        }
    
    async def stop(self, caller_context: IState) -> ActionResult:
        self._start_time = None
        self._end_time = None
        self._time_freq = None
        
        return ActionResult(value="OK", success=True)
    
    async def live(self, *, is_live: bool, caller_context: IState) -> ActionResult:
        self._live = is_live
        return ActionResult(value="OK", success=True)
    
    def reset(self, start: str, end: str, freq: str) -> None:
        timestep_to_freq = {
            'Tick': timedelta(microseconds=0),
            'S1': timedelta(seconds=1),
            'M1': timedelta(minutes=1),
            'M5': timedelta(minutes=5),
            'M10': timedelta(minutes=10),
            'M15': timedelta(minutes=15),
            'M20': timedelta(minutes=20),
            'M30': timedelta(minutes=30),
            'H1': timedelta(hours=1),
            'H4': timedelta(hours=4),
            'D1': timedelta(days=1),
        }
        self._time_freq_str = freq
        if freq not in timestep_to_freq:
            raise ValueError(f"Invalid frequency {freq}")
        
        freq_td: timedelta = timestep_to_freq[freq]
        
        try:
            start_dt = self._parse_dt(start)
            end_dt = self._parse_dt(end)
        except ValueError as e:
            logger.error(e)
            raise e
            
        logger.info(f"Replaying data from {start_dt} to {end_dt} with frequency {freq}")
        
        self._start_time, self._end_time, self._time_freq = start_dt, end_dt, freq_td
        self._time = start_dt
        self._live = False
        
        if freq_td >= timedelta(minutes=1):
            self._start_time = self._start_time.replace(second=0)
            self._end_time = self._end_time.replace(second=0)
        if freq_td >= timedelta(hours=1):
            self._start_time = self._start_time.replace(minute=0)
            self._end_time = self._end_time.replace(minute=0)
        if freq_td >= timedelta(days=1):
            self._start_time = self._start_time.replace(hour=0)
            self._end_time = self._end_time.replace(hour=0)
    
    async def replay(self, *, start: str, end: str, freq: str, caller_context: IState) -> ActionResult:
        
        try:
            self.reset(start, end, freq)
        except ValueError as e:
            return ActionResult(value=str(e), success=False)
        
        return ActionResult(value="OK", success=True)
    
    def _parse_dt(self, dt_str: str) -> datetime:
        try:
            return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=UTC)
        except ValueError as e1:
            logger.error(e1)
            try:
                return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
            except ValueError as e2:
                logger.error(e2)
                try:
                    return datetime.strptime(dt_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59).replace(tzinfo=UTC)
                except ValueError as e3:
                    logger.error(e3)
                    raise ValueError(f"Invalid date format {dt_str}")
                
    def get_time(self) -> datetime:
        if self._live:
            return datetime.now(UTC)
        return self._time
    
    def ask(self, symbol: str, timestamp: datetime | None = None) -> float:
        return self.price(symbol, 'ask', timestamp)

    def bid(self, symbol: str, timestamp: datetime | None = None) -> float:
        return self.price(symbol, 'bid', timestamp)
    
    def price(self, symbol: str, price_type: Literal['ask', 'bid'], timestamp: datetime | None = None) -> float:
        """
        Returns the ask or bid price for the symbol at the given timestamp
        Returns -1 if the price is not available
        """
        assert symbol in self._data, f"Symbol {symbol} not loaded"
        if timestamp is None:
            timestamp = self.get_time()
        if self._data[symbol]['source'] == 'pd.dataframe':
            if timestamp in self._data[symbol]['dict']:
                return self._data[symbol]['dict'][timestamp][price_type]
            else:
                
                idx = bisect_left(self._data[symbol]['index'], timestamp) - 1
                
                if idx < 0:
                    return -1

                loc = self._data[symbol]['index'][idx]
                
                return self._data[symbol]['dict'][loc][price_type]
            
        raise NotImplementedError
    
    def ticks(self, symbol: str, timestamp: datetime | None, max_ticks: int | None = 1) -> pd.DataFrame:
        assert symbol in self._data, f"Symbol {symbol} not loaded"
        
        if self._data[symbol]['source'] == 'pd.dataframe':
            if timestamp is None:
                df = self._data[symbol]['df']
            else:
                df = self._data[symbol]['df'].loc[:timestamp + timedelta(microseconds=1)]
            if max_ticks is not None:
                return df.iloc[-max_ticks:]
            else:
                return df
        raise NotImplementedError
    
    def bars(self, symbol: str, timeframe: str, timestamp: datetime | None = None, max_bars: int | None = None) -> pd.DataFrame:
        if (symbol, timeframe, timestamp) not in self._cache:
            self._cache[(symbol, timeframe, timestamp)] = aggregate_prices(self.ticks(symbol, timestamp, None), timeframe)
        df = self._cache[(symbol, timeframe, timestamp)]
        if max_bars is not None:
            df = df.iloc[-max_bars:]
        return df
    
    def next_time(self) -> Optional[datetime]:
        
        if self._time > self._end_time:
            return None
        if self._time.weekday() in (5,6):
            return self._time + timedelta(days=1)
        else:
            return self._time + self._time_freq
    
    def step(self) -> None:
        if self._time_freq is not None and self._time is not None:
            self._time = self.next_time()
        else:
            self._time = None
    
    async def tick(self) -> None:
        self.step()
    
    async def observe(self, caller_context: IState) -> IState:
        state = State()
        
        state.set_item('time', self._time.strftime("%Y-%m-%dT%H:%M:%S.000") if self._time is not None else None)
        state.set_item('live', self._live)
        state.set_item('is_on', (not self._live and self._time is not None) or self._live)
        state.set_item('start_time', self._start_time.strftime("%Y-%m-%dT%H:%M:%S.000") if self._start_time is not None else None)
        state.set_item('end_time', self._end_time.strftime("%Y-%m-%dT%H:%M:%S.000") if self._end_time is not None else None)
        state.set_item('freq', self._time_freq_str)
        
        return state
    
    def info(self) -> EntityInfo:
        return EntityInfo(
            name=self.__class__.__name__, 
            version="0.1.0", 
            params={}
        )
        