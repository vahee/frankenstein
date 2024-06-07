from typing import Any, Dict, Tuple
from agentopy import IAction, IState, IPolicy, WithActionSpaceMixin, EntityInfo, SharedStateKeys
from frankenstein.lib.trading.schemas import Signal


import logging

logger = logging.getLogger(__name__)

class TradingPolicy(WithActionSpaceMixin, IPolicy):

    async def action(self, state: IState) -> Tuple[IAction, Dict[str, Any], Dict[str, Any]]:
        
        caller_context = state.slice_by_prefix(SharedStateKeys.AGENT_ACTION_CONTEXT)
        
        signal: Signal = state.get_item('environment.components.SignalProvider.signal')
        
        if signal is None:
            return self.action_space.get_action('hold'), {"caller_context": caller_context}, {}
        
        tp = state.get_item('environment.components.ConfigProvider.tp')
        sl = state.get_item('environment.components.ConfigProvider.sl')
        lot_size = state.get_item('environment.components.ConfigProvider.lot_size')
        long_open_threshold = state.get_item('environment.components.ConfigProvider.long_open_threshold')
        long_close_threshold = state.get_item('environment.components.ConfigProvider.long_close_threshold')
        short_open_threshold = state.get_item('environment.components.ConfigProvider.short_open_threshold')
        short_close_threshold = state.get_item('environment.components.ConfigProvider.short_close_threshold')
        symbol = state.get_item('environment.components.ConfigProvider.symbol')
        
        try:
            assert tp is not None, "Take profit is not set"
            assert sl is not None, "Stop loss is not set"
            assert lot_size is not None, "Lot size is not set"
            assert long_open_threshold is not None, "Long open threshold is not set"
            assert long_close_threshold is not None, "Long close threshold is not set"
            assert short_open_threshold is not None, "Short open threshold is not set"
            assert short_close_threshold is not None, "Short close threshold is not set"
            assert symbol is not None, "Symbol is not set"
        except AssertionError:
            return self.action_space.get_action('hold'), {"caller_context": caller_context}, {}
        
        
        tp_pips = signal.tp_pips if signal.tp_pips is not None else tp
        sl_pips = signal.sl_pips if signal.sl_pips is not None else sl

        existing_positions = state.get_item(
            'environment.components.Broker.positions').get(symbol, None)

        if existing_positions is not None and existing_positions['is_open']:
            if existing_positions['is_long']:
                if signal.direction <= -long_close_threshold:
                    return self.action_space.get_action('close'), {
                        'symbol': symbol,
                        'comment': signal.comment,
                        "caller_context": caller_context
                    }, {}
            else:
                if signal.direction >= short_close_threshold:
                    return self.action_space.get_action('close'), {
                        'symbol': symbol,
                        'comment': signal.comment,
                        "caller_context": caller_context
                    }, {}
        else:
            if signal.direction >= long_open_threshold:
                return self.action_space.get_action('open'), {
                    'symbol': symbol,
                    'price': 0,
                    'volume': lot_size,
                    'is_long': True,
                    'take_profit_pips': tp_pips,
                    'stop_loss_pips': sl_pips,
                    'comment': signal.comment,
                    "caller_context": caller_context
                }, {}
            elif signal.direction <= -short_open_threshold:
                return self.action_space.get_action('open'), {
                    'symbol': symbol,
                    'price': 0,
                    'volume': lot_size,
                    'is_long': False,
                    'take_profit_pips': tp_pips,
                    'stop_loss_pips': sl_pips,
                    'comment': signal.comment,
                    "caller_context": caller_context
                }, {}
        return self.action_space.get_action('hold'), {"caller_context": caller_context}, {}
    
    def info(self) -> EntityInfo:
        return EntityInfo(
            name=self.__class__.__name__,
            version="0.1.0",
            params={}
        )