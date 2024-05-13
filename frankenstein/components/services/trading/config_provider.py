from agentopy import IEnvironmentComponent, IState, WithActionSpaceMixin, State, EntityInfo, ActionResult, Action

class ConfigProvider(WithActionSpaceMixin, IEnvironmentComponent):
    def __init__(self) -> None:
        super().__init__()
        
        # defaults
        self._params = {
            'symbol': 'EURUSD',
            'lot_size': 0.1,
            'open_threshold': 100,
            'close_threshold': 50,
            'sl': 100,
            'tp': 300
        }
        
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
                        'name': 'open_threshold', 
                        'type': 'number'
                    },
                    {
                        'name': 'close_threshold', 
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