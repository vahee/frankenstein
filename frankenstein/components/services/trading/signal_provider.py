from datetime import timedelta
from ta import volatility, momentum

from agentopy import IEnvironmentComponent, IState, WithActionSpaceMixin, State, EntityInfo, ActionResult, Action

from frankenstein.lib.trading.schemas import Signal
from frankenstein.lib.trading.protocols import IDataProvider


class SignalProvider(WithActionSpaceMixin, IEnvironmentComponent):
    def __init__(self, data_provider: IDataProvider, symbol: str) -> None:
        super().__init__()
        self._data_provider = data_provider
        
        self._prepared = False
        self._params = {}
        
        self.symbol = symbol
        
        self.rsi_bars_dict = {}
        self.bands_bars_dict = {}
        
        self._prepare(
            bands_timeframe='M10', 
            bands_window=7, 
            bands_dev=2, 
            rsi_timeframe='M10', 
            rsi_period=13
        )
        
        self._last_signal = Signal(self._data_provider.get_time(), 0, None, None, 'No signal', self.symbol)
        
        self.action_space.register_actions([
            Action('signal_provider_setup', "Sets the parameters", self.setup, self.info()),
        ])  
    
    async def setup(self, *, bands_timeframe: str, bands_window: int, bands_dev: int, rsi_timeframe: str, rsi_period: int, caller_context: IState) -> ActionResult:
        self._prepare(bands_timeframe, bands_window, bands_dev, rsi_timeframe, rsi_period)
        return ActionResult(value="OK", success=True)
    
    def _prepare(self, bands_timeframe: str, bands_window: int, bands_dev: int, rsi_timeframe: str, rsi_period: int) -> None:
        
        self._params = {
            'bands_timeframe': bands_timeframe,
            'bands_window': bands_window,
            'bands_dev': bands_dev,
            'rsi_timeframe': rsi_timeframe,
            'rsi_period': rsi_period,
        }
        
        # bands
        
        bands_bars = self._data_provider.bars(self.symbol, bands_timeframe)
        
        bands_bars['hband'] = volatility.bollinger_hband(
            bands_bars['close'], window=int(bands_window), window_dev=int(bands_dev))
        bands_bars['lband'] = volatility.bollinger_lband(
            bands_bars['close'], window=int(bands_window), window_dev=int(bands_dev))
        
        bands_bars['mband'] = volatility.bollinger_mavg(
            bands_bars['close'], window=int(bands_window))
        
        self.bands_bars_dict = bands_bars.to_dict('index')
        
        # rsi
        
        rsi_bars = self._data_provider.bars(self.symbol, rsi_timeframe)
        
        rsi_bars['rsi'] = momentum.rsi(rsi_bars['close'], window=int(rsi_period))
        
        self.rsi_bars_dict = rsi_bars.to_dict('index')
        
        self._last_signal = Signal(self._data_provider.get_time(), 0, None, None, 'No signal', self.symbol)
        self._prepared = True
    
    async def tick(self) -> None:
        try:
            timestamp = self._data_provider.get_time()
            assert timestamp is not None
            assert self._prepared
            
            timestamp = timestamp.replace(microsecond=0, second=0) - timedelta(minutes=1)
            hband = self.bands_bars_dict[timestamp]['hband']
            lband = self.bands_bars_dict[timestamp]['lband']
            mband = self.bands_bars_dict[timestamp]['mband']
            
            rsi = self.rsi_bars_dict[timestamp]['rsi']
            pass
        except (AssertionError, KeyError) as e:
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
            comment = f'Short, because bid > h, bid: {bid}, h: {hband}, bar_ts: {timestamp}'
        elif bid > mband:
            direction -= 50
            comment = f'Short, because bid > m, bid: {bid}, m: {mband}, bar_ts: {timestamp}'
        
        if bid < lband:
            direction += 100
            comment = f'Long, because bid < l, bid: {bid}, l: {lband}, bar_ts: {timestamp}'
        elif bid < mband:
            direction += 50
            comment = f'Long, because bid < m, bid: {bid}, m: {mband}, bar_ts: {timestamp}'
            
            
        direction += 100 - 2 * rsi
        
        direction /= 2
        comment = f'Because rsi: {rsi}, bar_ts: {timestamp}'
        
        self._last_signal = Signal(timestamp, direction, None, None, comment, self.symbol)
        
    
    async def observe(self, caller_context: IState) -> IState:
        state = State()
        state.set_item('signal', self._last_signal)
        state.set_item('symbol', self.symbol)
        state.set_item('params', self._params)
        return state
    
    def info(self) -> EntityInfo:
        return EntityInfo(
            'SignalProvider', 
            {
                'symbol': self.symbol,
                'params': self._params,
            }, 
        '0.1.0')