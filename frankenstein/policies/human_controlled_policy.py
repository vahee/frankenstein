from typing import Any, Dict, Tuple
from agentopy import IAction, IState, IPolicy, WithActionSpaceMixin, EntityInfo, SharedStateKeys, ActionResult, Action
import logging

logger = logging.getLogger(__name__)

class HumanControlledPolicy(WithActionSpaceMixin, IPolicy):
    
    def __init__(self):
        super().__init__()
        self.action_space.register_actions([
            Action(
                "nothing", "do nothing", self.do_nothing, self.info()),
        ])
        
    async def do_nothing(self, *, caller_context: IState) -> ActionResult:
        """Does nothing"""
        return ActionResult(value="Nothing to do", success=True)

    async def action(self, state: IState) -> Tuple[IAction, Dict[str, Any], Dict[str, Any]]:

        action_name = state.get_item('agent.components.RemoteControl.force_action.name')
        action_args = state.get_item('agent.components.RemoteControl.force_action.args')
        
        state.remove_item('agent.components.RemoteControl.force_action.name')
        state.remove_item('agent.components.RemoteControl.force_action.args')
        
        caller_context = state.slice_by_prefix(SharedStateKeys.AGENT_ACTION_CONTEXT)
        
        if not action_name:
            return self.action_space.get_action('nothing'), {"caller_context": caller_context}, {}
        
        action_args['caller_context'] = caller_context
        
        return self.action_space.get_action(action_name), action_args, {}
    
    def info(self) -> EntityInfo:
        return EntityInfo(
            name=self.__class__.__name__,
            version="0.1.0",
            params={}
        )