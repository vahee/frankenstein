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
            bands_enabled=True,
            bands_timeframe='M10', 
            bands_window=7, 
            bands_dev=2, 
            rsi_enabled=True,
            rsi_timeframe='M10', 
            rsi_period=13,
            stochastic_enabled=True,
            stochastic_timeframe='M10',
            stochastic_smooth=3,
            stochastic_period=14
        )
        
        self._last_signal = Signal(self._data_provider.get_time(), 0, None, None, 'No signal', self.symbol)
        
        self.action_space.register_actions([
            Action('signal_provider_setup', "Sets the parameters", self.setup, self.info()),
        ])  
    
    async def setup(self, 
        *,
        bands_enabled: bool,
        bands_timeframe: str, 
        bands_window: int, 
        bands_dev: int, 
        rsi_enabled: bool,
        rsi_timeframe: str, 
        rsi_period: int,
        stochastic_enabled: bool,
        stochastic_timeframe: str,
        stochastic_smooth: int,
        stochastic_period: int,
        caller_context: IState
    ) -> ActionResult:
        self._prepare(
            bands_enabled=bands_enabled,
            bands_timeframe=bands_timeframe, 
            bands_window=bands_window, 
            bands_dev=bands_dev, 
            rsi_enabled=rsi_enabled,
            rsi_timeframe=rsi_timeframe, 
            rsi_period=rsi_period,
            stochastic_enabled=stochastic_enabled,
            stochastic_timeframe=stochastic_timeframe,
            stochastic_period=stochastic_period,
            stochastic_smooth=stochastic_smooth
        )
        return ActionResult(value="OK", success=True)
    
    def _prepare(self, 
        *,
        bands_enabled: bool,
        bands_timeframe: str, 
        bands_window: int, 
        bands_dev: int, 
        rsi_enabled: bool,
        rsi_timeframe: str, 
        rsi_period: int,
        stochastic_enabled: bool,
        stochastic_timeframe: str,
        stochastic_period: int,
        stochastic_smooth: int,
    ) -> None:
        
        self._params = {
            'bands_enabled': bands_enabled,
            'bands_timeframe': bands_timeframe,
            'bands_window': bands_window,
            'bands_dev': bands_dev,
            'rsi_enabled': rsi_enabled,
            'rsi_timeframe': rsi_timeframe,
            'rsi_period': rsi_period,
            'stochastic_enabled': stochastic_enabled,
            'stochastic_timeframe': stochastic_timeframe,
            'stochastic_period': stochastic_period,
            'stochastic_smooth': stochastic_smooth,
        }
        
        # bands
        if bands_enabled:
            bands_bars = self._data_provider.bars(self.symbol, bands_timeframe)
            
            bands_bars['hband'] = volatility.bollinger_hband(
                bands_bars['close'], window=int(bands_window), window_dev=int(bands_dev))
            bands_bars['lband'] = volatility.bollinger_lband(
                bands_bars['close'], window=int(bands_window), window_dev=int(bands_dev))
            
            bands_bars['mband'] = volatility.bollinger_mavg(
                bands_bars['close'], window=int(bands_window))
            
            self.bands_bars_dict = bands_bars.to_dict('index')
        
        # rsi
        if rsi_enabled:
            rsi_bars = self._data_provider.bars(self.symbol, rsi_timeframe)
            
            rsi_bars['rsi'] = momentum.rsi(rsi_bars['close'], window=int(rsi_period))
            
            self.rsi_bars_dict = rsi_bars.to_dict('index')
        
        # stochastic
        
        if stochastic_enabled:
            stochastic_bars = self._data_provider.bars(self.symbol, stochastic_timeframe)
            
            stochastic_bars['stoch'] = momentum.stoch(stochastic_bars['high'], stochastic_bars['low'], stochastic_bars['close'], window=int(stochastic_period), smooth_window=int(stochastic_smooth))
            
            self.stochastic_bars_dict = stochastic_bars.to_dict('index')
        
        self._last_signal = Signal(self._data_provider.get_time(), 0, None, None, 'No signal', self.symbol)
        self._prepared = True
    
    async def tick(self) -> None:
        stochastic = 0
        rsi = 0
        hband = 0
        lband = 0
        mband = 0
        
        try:
            timestamp = self._data_provider.get_time()
            assert timestamp is not None
            assert self._prepared
            
            timestamp = timestamp.replace(microsecond=0, second=0)
            
            if self._params['bands_enabled']:
                
                hband = self.bands_bars_dict[timestamp]['hband']
                lband = self.bands_bars_dict[timestamp]['lband']
                mband = self.bands_bars_dict[timestamp]['mband']
                
            if self._params['rsi_enabled']:
                
                rsi = self.rsi_bars_dict[timestamp]['rsi']
            
            if self._params['stochastic_enabled']:
                stochastic = self.stochastic_bars_dict[timestamp]['stoch']
            
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
        
        n_indicators = 0
        direction = 0
        
        if self._params['bands_enabled']:
            n_indicators += 1
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
        
        if self._params['rsi_enabled']:
            n_indicators += 1
            direction += 100 - 2 * rsi    
        
        if self._params['stochastic_enabled']:
            n_indicators += 1
            direction += 100 - 2 * stochastic
        
        if n_indicators == 0:
            self._last_signal = Signal(timestamp, 0, None, None, 'No signal', self.symbol)
            return
        
        direction /= n_indicators
        comment = f'bid: {bid}, rsi: {rsi}, stochastic: {stochastic}, mband: {mband}, hband: {hband}, lband: {lband}, bar_ts: {timestamp}'
        
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