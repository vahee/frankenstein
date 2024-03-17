from datetime import datetime, timedelta, UTC
from typing import Tuple, Optional, Literal
from bisect import bisect_left
import pandas as pd

from agentopy import IEnvironmentComponent, WithStateMixin, WithActionSpaceMixin
from frankenstein.lib.trading.protocols import IDataProvider
from frankenstein.lib.trading.utils import aggregate_ticks


class DataProvider(WithStateMixin, WithActionSpaceMixin, IDataProvider, IEnvironmentComponent):
    def __init__(self, *, time_range: Optional[Tuple[datetime, datetime, timedelta]]) -> None:
        super().__init__()

        if time_range is not None:
            start, end, freq = time_range
            
            assert isinstance(start, datetime), "time_range[0] must be datetime"
            assert isinstance(end, datetime), "time_range[1] must be datetime"
            assert isinstance(freq, timedelta), "time_range[2] must be timedelta"
            assert start < end, "time_range[0] must be less than time_range[1]"
            assert freq >= timedelta(seconds=1), "time_range[2] must be at least 1 second"
                
            self._start_time, self._end_time, self._time_freq = start, end, freq
            
        else:
            self.state.set_item('time', datetime.now(UTC))
        
        self._cache = {}
        self._data = {}
        
    def load_ticks_dataframe(self, df: pd.DataFrame, symbol: str):
        self._data[symbol] = {
            'index': df.index.tolist(),
            'df': df,
            'dict': df.to_dict('index'),
            'source': 'dataframe'
        }
        
    def get_time(self) -> datetime:
        return self.state.get_item('time')
    
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
        if self._data[symbol]['source'] == 'dataframe':
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
        
        if self._data[symbol]['source'] == 'dataframe':
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
            self._cache[(symbol, timeframe, timestamp)] = aggregate_ticks(self.ticks(symbol, timestamp, None), timeframe)
        df = self._cache[(symbol, timeframe, timestamp)]
        if max_bars is not None:
            df = df.iloc[-max_bars:]
        return df
    
    def _next_time(self) -> Optional[datetime]:
        if self.state.get_item('time') > self._end_time:
            return None
        if self.state.get_item('time').weekday() in (5,6):
            return self.state.get_item('time') + timedelta(days=1)
        else:
            return self.state.get_item('time') + self._time_freq
    
    async def on_tick(self) -> None:
        if self._time_freq is not None:
            if self.state.get_item('time') is None:
                self.state.set_item('time', self._start_time)
            else:
                self.state.set_item('time', self._next_time())
        else:
            self.state.set_item('time', datetime.now(UTC))
        
        