from datetime import timedelta
from ta import volatility

from agentopy import IEnvironmentComponent, WithStateMixin, WithActionSpaceMixin

from frankenstein.lib.trading.schemas import Signal
from frankenstein.lib.trading.protocols import IDataProvider

class BandsSignalProvider(WithStateMixin, WithActionSpaceMixin, IEnvironmentComponent):
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

    async def on_tick(self) -> None:
        
        try:
            timestamp = self._data_provider.get_time()
            assert timestamp is not None
            
            timestamp = timestamp.replace(microsecond=0, second=0) - timedelta(minutes=1)
            hband = self.bars_dict[timestamp]['hband']
            lband = self.bars_dict[timestamp]['lband']
            mband = self.bars_dict[timestamp]['mband']
        except (AssertionError, KeyError):
            self.state.set_item('signal', Signal(timestamp, 0, None, None, 'No signal', self.symbol))
            return
        
        
        direction = 0
        bid = self._data_provider.bid(self.symbol)
        if bid is None:
            self.state.set_item('signal', Signal(timestamp, 0, None, None, 'No signal', self.symbol))
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
        
        
        self.state.set_item('signal', Signal(timestamp, direction, None, None, comment, self.symbol))