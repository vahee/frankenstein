from datetime import timedelta
from ta import volatility
import asyncio as aio

from agentopy import IEnvironmentComponent, IState, WithActionSpaceMixin, State, EntityInfo

from frankenstein.lib.trading.schemas import Signal
from frankenstein.lib.trading.protocols import IDataProvider


class BandsSignalProvider(WithActionSpaceMixin, IEnvironmentComponent):
    def __init__(self, data_provider: IDataProvider, symbol: str, timeframe: str, bands_window: int, bands_dev: int) -> None:
        super().__init__()
        self._data_provider = data_provider
        
        self.symbol = symbol
        self.bars = self._data_provider.bars(symbol, timeframe)
        
        self.bars['hband'] = volatility.bollinger_hband(
            self.bars['close'], window=bands_window, window_dev=bands_dev)
        self.bars['lband'] = volatility.bollinger_lband(
            self.bars['close'], window=bands_window, window_dev=bands_dev)
        
        self.bars['mband'] = volatility.bollinger_mavg(
            self.bars['close'], window=bands_window)
        
        self.bars_dict = self.bars.to_dict('index')
        self._last_signal = Signal(self._data_provider.get_time(), 0, None, None, 'No signal', self.symbol)

    async def tick(self) -> None:
        try:
            timestamp = self._data_provider.get_time()
            assert timestamp is not None
            
            timestamp = timestamp.replace(microsecond=0, second=0) - timedelta(minutes=1)
            hband = self.bars_dict[timestamp]['hband']
            lband = self.bars_dict[timestamp]['lband']
            mband = self.bars_dict[timestamp]['mband']
        except (AssertionError, KeyError):
            self._last_signal = Signal(timestamp, 0, None, None, 'No signal', self.symbol)
            return
        
        
        direction = 0
        bid = self._data_provider.bid(self.symbol)
        if bid is None:
            self._last_signal = Signal(timestamp, 0, None, None, 'No signal', self.symbol)
            return
        
        comment = 'Action'
        if bid > hband:
            direction -= 100
            comment = f'Short, because bid > h, bid: {bid}, h: {hband}, m1_ts: {timestamp}'
        elif bid > mband:
            direction -= 50
            comment = f'Short, because bid > m, bid: {bid}, m: {mband}, m1_ts: {timestamp}'
        
        if bid < lband:
            direction += 100
            comment = f'Long, because bid < l, bid: {bid}, l: {lband}, m1_ts: {timestamp}'
        elif bid < mband:
            direction += 50
            comment = f'Long, because bid < m, bid: {bid}, m: {mband}, m1_ts: {timestamp}'
        
        self._last_signal = Signal(timestamp, direction, None, None, comment, self.symbol)
        
    
    async def observe(self, caller_context: IState) -> IState:
        state = State()
        state.set_item('signal', self._last_signal)
        return state
    
    def info(self) -> EntityInfo:
        return EntityInfo(
            'BandsSignalProvider', 
            {
                'symbol': self.symbol
            }, 
        '0.1.0')