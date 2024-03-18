from typing import Any, Dict, Tuple
from agentopy import IAction, IState, IPolicy, WithActionSpaceMixin


class ManagerPolicy(WithActionSpaceMixin, IPolicy):

    async def action(self, state: IState) -> Tuple[IAction, Dict[str, Any], Dict[str, Any]]:

        action_name = state.get_item('agent/components/remote_control/force_action/name')
        action_args = state.get_item('agent/components/remote_control/force_action/args')
        
        if not action_name:
            return self.action_space.get_action('nothing'), {}, {}
        
        return self.action_space.get_action(action_name), action_args, {}