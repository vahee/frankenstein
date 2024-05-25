import time
import asyncio as aio
from agentopy import IEnvironmentComponent, IState, WithActionSpaceMixin, Action, EntityInfo, State, ActionResult
from frankenstein.lib.trading.protocols import IDataProvider


class Broker(WithActionSpaceMixin, IEnvironmentComponent):
    def __init__(self, data_provider: IDataProvider) -> None:
        super().__init__()
        self.start_time = time.time()
        
        self._data_provider = data_provider

        self.action_space.register_actions(
            [
                Action('open', "Opens a position", self.open, self.info()),
                Action('hold', "Holds the position", self.hold, self.info()),
                Action('close', "Closes the position", self.close, self.info()),
                Action('broker_set_balance', "Sets the balance", self.set_balance, self.info()),
                Action('broker_is_live', "Sets the live mode", self.is_live, self.info()),
                Action('broker_is_on', "Sets the broker on/off", self.is_on, self.info()),
                Action('broker_prepare_account', "Sets the broker parameters", self.prepare_account, self.info()),
            ]
        )
        
        self._is_on = False
        self._is_live = False
        
        self._last_ask = None
        self._last_bid = None
        self._balance = 0
        self._leverage = 0
        self._point = 0
        self._lot_in_units = 0
        self._equity = 0
        self._pl = 0
        self._positions = {}
        self._trades = []
        self._total_trade_count = 0
        
        self.set_params()
        self.reset()
        

    async def set_balance(self, *, balance: float, caller_context: IState) -> ActionResult:
        self._balance = int(balance)
        return ActionResult(value="OK", success=True)
    
    async def is_live(self, *, is_live: bool, caller_context: IState) -> ActionResult:
        self._is_live = is_live
        return ActionResult(value="OK", success=True)
    
    async def is_on(self, *, is_on: bool, caller_context: IState) -> ActionResult:
        self._is_on = is_on
        return ActionResult(value="OK", success=True)
    
    async def prepare_account(self, *, balance: float, leverage: int, point: float, lot_in_units: int, caller_context: IState) -> ActionResult:
        
        self.reset()
        self.set_params(balance, leverage, point, lot_in_units)
        
        return ActionResult(value="OK", success=True)
    
    def reset(self) -> None:
        self._trades = []
        self._positions = {}
        self._pl = 0
        self._equity = self._balance
        self._total_trade_count = 0
        self._last_ask = None
        self._last_bid = None
    
    def set_params(self, balance: float = 10000, leverage: int = 30, point: float = 1, lot_in_units: int = 1) -> None:
        self._balance = float(balance)
        self._leverage = float(leverage)
        self._point = float(point)
        self._lot_in_units = float(lot_in_units)
    
    async def tick(self) -> None:
        timestamp = self._data_provider.get_time()
        if timestamp is None:
            return 
        ask = self._data_provider.ask('EURUSD')
        bid = self._data_provider.bid('EURUSD')
        
        ask = ask if ask is not None else self._last_ask
        bid = bid if bid is not None else self._last_bid
        
        self._last_ask = ask
        self._last_bid = bid
        
        if ask is None or bid is None:
            raise Exception("Ask or bid is None")
        
        ask = round(ask, 5)
        bid = round(bid, 5)
    
        positions = self._positions.copy()
        
        try:

            for symbol, position in positions.items():
                if not position['is_open']:
                    continue

                pl = bid - \
                    position['price'] if position['is_long'] else position['price'] - ask

                position['pl'] = pl

                self._equity = self._balance + \
                    position['volume'] * position['pl'] * self._lot_in_units

                if pl > position['take_profit_pips'] * self._point or pl < -position['stop_loss_pips'] * self._point:
                    state = State()
                    await self.close(symbol=symbol, comment="TP/SL reached", caller_context=state)
        except Exception as e:
            print(e)
            raise e

    async def hold(self, *, caller_context: IState) -> None:
        ...

    async def open(self, *, symbol: str, price: float, volume: float, is_long: bool, take_profit_pips: int, stop_loss_pips: int, comment: str, caller_context: IState) -> ActionResult:
        if not self._is_on:
            return ActionResult(value="Broker is off", success=False)
        assert self._last_ask is not None and self._last_bid is not None, "Ask or bid is None"
        self._positions[symbol] = {
            'price': price,
            'volume': volume,
            'is_long': is_long,
            'take_profit_pips': take_profit_pips,
            'stop_loss_pips': stop_loss_pips,
            'pl': self._last_bid - price if is_long else price - self._last_ask,
            'is_open': True,
            'open_timestamp': self._data_provider.get_time(),
            'open_comment': comment
        }
        self._total_trade_count += 1
        self._trades.append(self._positions[symbol])
        return ActionResult(value="OK", success=True)

    async def close(self, *, symbol: str, comment: str, caller_context: IState) -> ActionResult:
        if not self._is_on:
            return ActionResult(value="Broker is off", success=False)
        
        position = self._positions.pop(symbol, None)
        
        position['close_timestamp'] = self._data_provider.get_time()
        position['close_comment'] = comment
        
        if position is not None:
            position['is_open'] = False
        
        self._pl += self._equity - self._balance
        self._balance = self._equity
        
        return ActionResult(value="OK", success=True)
        
    async def observe(self, caller_context: IState) -> IState:
        state = State()
        state.set_item('positions', self._positions)
        state.set_item('ask', self._last_ask)
        state.set_item('bid', self._last_bid)
        state.set_item('balance', self._balance)
        state.set_item('equity', self._equity)
        state.set_item("pl", self._pl)
        state.set_item('leverage', self._leverage)
        state.set_item('point', self._point)
        state.set_item('total_trade_count', self._total_trade_count)
        state.set_item('trades', self._trades)
        state.set_item('is_live', self._is_live)
        state.set_item('is_on', self._is_on)
        state.set_item('lot_in_units', self._lot_in_units)
        return state
    
    
    def info(self) -> EntityInfo:
        return EntityInfo(
            name=self.__class__.__name__,
            version="0.1.0",
            params={}
        )
        