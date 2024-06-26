from typing import Dict, Any
from agentopy import IEnvironmentComponent, IState, WithActionSpaceMixin, State, EntityInfo, ActionResult, Action

class ConfigProvider(WithActionSpaceMixin, IEnvironmentComponent):
    def __init__(self) -> None:
        super().__init__()
        
        # defaults
        self._params = {
            'symbol': 'EURUSD',
            'lot_size': 0.1,
            'long_open_threshold': 50,
            'long_close_threshold': 30,
            'short_open_threshold': 50,
            'short_close_threshold': 30,
            'sl': 100,
            'tp': 300
        }
        
        self._status: Dict[str, Any] = dict()
        
        self.action_space.register_actions([
            Action('config_provider_set_params', "Sets the parameters", self.set_params, self.info()),
        ])
        
    async def set_params(self, *, caller_context: IState, **kwargs) -> ActionResult:
        
        self._params = {}
        accepted_params = {param["name"]: param["type"] for param in self.info().params.get("params", [])}
        for key, value in kwargs.items():
            if key in accepted_params:
                if accepted_params[key] == 'number':
                    value = float(value)
                self._params[key] = value
        
        return ActionResult(value="OK", success=True)
    
    async def tick(self) -> None:
        ...
    
    async def observe(self, caller_context: IState) -> IState:
        state = State()
        
        for key, value in self._params.items():
            state.set_item(key, value)
        
        state.set_item('status', self._status)
        return state
    
    def info(self) -> EntityInfo:
        return EntityInfo(
            'ConfigProvider', 
            {
                'params': [
                    {
                        'name': 'symbol', 
                        'type': 'text',
                    },
                    {
                        'name': 'lot_size', 
                        'type': 'number'
                    },
                    {
                        'name': 'long_open_threshold', 
                        'type': 'number'
                    },
                    {
                        'name': 'long_close_threshold', 
                        'type': 'number'
                    },
                    {
                        'name': 'short_open_threshold', 
                        'type': 'number'
                    },
                    {
                        'name': 'short_close_threshold', 
                        'type': 'number'
                    },
                    {
                        'name': 'sl', 
                        'type': 'number'
                    },
                    {
                        'name': 'tp', 
                        'type': 'number'
                    }
                        
                ]
            }, 
        '0.1.0')