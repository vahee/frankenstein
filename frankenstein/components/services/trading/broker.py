import time
from agentopy import IEnvironmentComponent, WithActionSpaceMixin, WithStateMixin, Action, EntityInfo
from frankenstein.lib.trading.protocols import IDataProvider


class Broker(WithStateMixin, WithActionSpaceMixin, IEnvironmentComponent):
    def __init__(self, data_provider: IDataProvider, max_state_len: int = 50) -> None:
        super().__init__(max_state_len)
        self.start_time = time.time()
        
        self._data_provider = data_provider

        self.action_space.register_actions(
            [
                Action('open', "Opens a position", self.open, self.info()),
                Action('hold', "Holds the position", self.hold, self.info()),
                Action('close', "Closes the position", self.close, self.info())
            ]
        )

        self._positions = {}
        self._balance = 10000
        self._equity = 10000
        self._leverage = 30
        self._point = 0.0001
        self._total_trade_count = 0

        self._lot_in_units = 100000
        
        self._trades = []

    async def on_tick(self) -> None:
        timestamp = self._data_provider.get_time()
        if timestamp is None:
            print(f"Balance: {self._balance}, Equity: {self._equity}, Leverage: {self._leverage}, Point: {self._point}, Total trades: {self._total_trade_count}")
            raise Exception("End of time")
        ask = self._data_provider.ask('EURUSD')
        bid = self._data_provider.bid('EURUSD')
        
        ask = ask if ask is not None else self.state.get_item('ask')
        bid = bid if bid is not None else self.state.get_item('bid')
        
        if ask is None or bid is None:
            raise Exception("Ask or bid is None")
        
        ask = round(ask, 5)
        bid = round(bid, 5)    
        
        self.state.set_item('ask', ask)
        self.state.set_item('bid', bid)
    
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
                    await self.close(symbol, "TP/SL reached")
        except Exception as e:
            print(e)
            raise e
        
        self.state.set_item('positions', self._positions)

    async def hold(self) -> None:
        ...

    async def open(self, symbol: str, price: float, volume: float, is_long: bool, take_profit_pips: int, stop_loss_pips: int, comment: str) -> None:
        self._positions[symbol] = {
            'price': price,
            'volume': volume,
            'is_long': is_long,
            'take_profit_pips': take_profit_pips,
            'stop_loss_pips': stop_loss_pips,
            'pl': self.state.get_item('bid') - price if is_long else price - self.state.get_item('ask'),
            'is_open': True,
            'timestamp': self._data_provider.get_time(),
            'comment': comment
        }
        self._total_trade_count += 1
        self._trades.append(self._positions[symbol])

    async def close(self, symbol: str, _: str) -> None:
        position = self._positions.pop(symbol, None)
        
        if position is not None:
            position['is_open'] = False
        self._balance = self._equity
        self.state.set_item('positions', self._positions)
        
    
    def info(self) -> EntityInfo:
        return EntityInfo(
            name=self.__class__.__name__,
            version="0.1.0",
            params={}
        )
        