from typing import Any, Dict, Tuple
from agentopy import IAction, IState, IPolicy, WithActionSpaceMixin, EntityInfo, SharedStateKeys
from frankenstein.lib.trading.schemas import Signal


import logging

logger = logging.getLogger(__name__)

class TradingPolicy(WithActionSpaceMixin, IPolicy):

    async def action(self, state: IState) -> Tuple[IAction, Dict[str, Any], Dict[str, Any]]:
        state_items = state._data
        action_name = state.get_item('agent.components.RemoteControl.force_action.name')
        action_args = state.get_item('agent.components.RemoteControl.force_action.args')
        
        state.remove_item('agent.components.RemoteControl.force_action.name')
        state.remove_item('agent.components.RemoteControl.force_action.args')
        
        caller_context = state.slice_by_prefix(SharedStateKeys.AGENT_ACTION_CONTEXT)
        
        if action_name:
            action_args['caller_context'] = caller_context
            return self.action_space.get_action(action_name), action_args, {}
        
        signal: Signal = state.get_item('environment.components.SignalProvider.signal')
        
        if signal is None:
            return self.action_space.get_action('hold'), {"caller_context": caller_context}, {}
        
        tp = state.get_item('environment.components.ConfigProvider.tp')
        sl = state.get_item('environment.components.ConfigProvider.sl')
        lot_size = state.get_item('environment.components.ConfigProvider.lot_size')
        open_threshold = state.get_item('environment.components.ConfigProvider.open_threshold')
        close_threshold = state.get_item('environment.components.ConfigProvider.close_threshold')
        symbol = state.get_item('environment.components.ConfigProvider.symbol')
        
        try:
            assert tp is not None, "Take profit is not set"
            assert sl is not None, "Stop loss is not set"
            assert lot_size is not None, "Lot size is not set"
            assert open_threshold is not None, "Open threshold is not set"
            assert close_threshold is not None, "Close threshold is not set"
            assert symbol is not None, "Symbol is not set"
        except AssertionError:
            return self.action_space.get_action('hold'), {"caller_context": caller_context}, {}
        
        
        tp_pips = signal.tp_pips if signal.tp_pips is not None else tp
        sl_pips = signal.sl_pips if signal.sl_pips is not None else sl

        existing_positions = state.get_item(
            'environment.components.Broker.positions').get(symbol, None)

        if existing_positions is not None and existing_positions['is_open']:
            if signal.direction <= -close_threshold:
                if existing_positions['is_long']:
                    return self.action_space.get_action('close'), {
                        'symbol': symbol,
                        'comment': signal.comment,
                        "caller_context": caller_context
                    }, {}
            elif signal.direction >= close_threshold:
                if not existing_positions['is_long']:
                    return self.action_space.get_action('close'), {
                        'symbol': symbol,
                        'comment': signal.comment,
                        "caller_context": caller_context
                    }, {}
        else:
            if signal.direction >= open_threshold:
                if existing_positions is not None and existing_positions['is_long'] and existing_positions['is_open']:
                    return self.action_space.get_action('hold'), {"caller_context": caller_context}, {}

                return self.action_space.get_action('open'), {
                    'symbol': symbol,
                    'price': state.get_item('environment.components.Broker.ask'),
                    'volume': lot_size,
                    'is_long': True,
                    'take_profit_pips': tp_pips,
                    'stop_loss_pips': sl_pips,
                    'comment': signal.comment,
                    "caller_context": caller_context
                }, {}
            elif signal.direction <= -open_threshold:
                if existing_positions is not None and not existing_positions['is_long'] and existing_positions['is_open']:
                    return self.action_space.get_action('hold'), {"caller_context": caller_context}, {}

                return self.action_space.get_action('open'), {
                    'symbol': symbol,
                    'price': state.get_item('environment.components.Broker.bid'),
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