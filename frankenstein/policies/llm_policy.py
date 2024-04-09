from typing import Tuple, Dict, Any, Callable, Optional
from datetime import datetime
from os import linesep
import orjson

from agentopy import IState, IAction, IPolicy, WithActionSpaceMixin, Action, ActionResult, EntityInfo, SharedStateKeys, IState

from frankenstein.lib.language.protocols import ILanguageModel


class LLMPolicy(WithActionSpaceMixin, IPolicy):
    """Implements a policy that uses a language model to generate actions"""

    def __init__(self,
                 language_model: ILanguageModel,
                 system_prompt: str,
                 response_template: str,
                 response_parser: Optional[Callable[[
                     Dict[str, Any]], Tuple[str, Dict[str, Any], Dict[str, Any]]]] = None,
                 wait_timeout_s: int = 300
                 ) -> None:
        super().__init__()
        self._language_model: ILanguageModel = language_model
        self._system_prompt: str = system_prompt
        self._response_template: str = response_template
        self._response_parser: Callable[[
            Dict[str, Any]], Tuple[str, Dict[str, Any], Dict[str, Any]]] = response_parser or self._default_response_parser

        self._wait_timeout_s: int = wait_timeout_s
        self._wait_start_ts = None

        self.action_space.register_actions([
            Action(
                "wait", "use this action to wait for a change in environment", self._wait, self.info()),
        ])

    async def _wait(self, *, caller_context: IState) -> ActionResult:
        if self._wait_start_ts is None:
            self._wait_start_ts = datetime.now()
            return ActionResult(value=f"Start waiting for a change in environment for {self._wait_timeout_s} seconds.", success=True)
        else:
            wait_time_left_s = self._wait_timeout_s - \
                (datetime.now() - self._wait_start_ts).seconds
            return ActionResult(value=f"Waiting for another {wait_time_left_s} seconds.", success=False)

    def _default_response_parser(self, response: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        action = response.pop("action")
        args = response.pop("args")

        thoughts = {}

        for k, v in response.items():
            thoughts[k.capitalize()] = v

        thoughts["Action"] = action

        if args:
            thoughts["Arguments"] = ', '.join(
                [f'{k}: {v}' for k, v in args.items()])
        return action, args, thoughts

    async def action(self, state: IState) -> Tuple[IAction, Dict[str, Any], Dict[str, Any]]:
        """Generates an action based on the current state of the environment."""
        action_name = state.get_item('agent/components/RemoteControl/force_action/name')
        action_args = state.get_item('agent/components/RemoteControl/force_action/args')
        
        state.remove_item('agent/components/RemoteControl/force_action/name')
        state.remove_item('agent/components/RemoteControl/force_action/args')
        
        caller_context = state.slice_by_prefix(SharedStateKeys.AGENT_ACTION_CONTEXT)
        
        if action_name:
            if isinstance(action_args, dict):
                action_args['caller_context'] = caller_context
            return self.action_space.get_action(action_name), action_args, {"Thoughts": "Forced action."}
    
        if self._wait_start_ts is not None:
            # TODO: check if there is an actual change in the environment and stop waiting if there is
            messenger_status = state.get_item(
                "environment/components/Messenger/status")

            if not messenger_status or not messenger_status.get("num_new_messages"):
                wait_time_left_s = self._wait_timeout_s - \
                    (datetime.now() - self._wait_start_ts).seconds
                if wait_time_left_s > 0:
                    return self.action_space.get_action("wait"), {"caller_context": caller_context}, {"Thoughts": "Waiting for a change in environment."}

        self._wait_start_ts = None

        state_text = self._format_input(state)

        query = f"""The following in triple quotes is the current state: \"\"\"{linesep}{state_text}{linesep}\"\"\""""

        response = await self._query_language_model(query, state)

        action_name, args, thoughts = self._response_parser(response)

        action = self.action_space.get_action(action_name)
        
        args['caller_context'] = caller_context

        return action, args, thoughts

    def _format_input(self, state: IState) -> str:
        result = []
        
        for key, value in state.items().items():
            if (key.startswith("agent/components") or key.startswith("environment/components")) and not key.endswith("__"):
                result.append(self._format_item(key, value))

        return linesep.join(result)

    def _format_item(self, key: str, value: str | Dict[str, str]) -> str:

        if isinstance(value, str):
            return f"""{key}: {value}{linesep}"""

        if isinstance(value, list):
            return f"""{key}: {', '.join(value)}{linesep}"""
            
        return f"""{key}: {linesep.join(
                        [f"{k}: {v}" for k, v in value.items()])}{linesep}"""

    def _create_language_model_query(self, query: str) -> str:
        """Creates a query for the language model"""
        return self._response_template.format(
            query=query,
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S, %A")
        )

    async def _query_language_model(self, query: str, state: IState) -> Dict[str, Any]:
        """Queries the language model with the specified query and returns the response as a dictionary"""
        query = self._create_language_model_query(query)
        actions = self.action_space.all_actions()
        f = "{}: {}"
        
        actions_list = []
        
        for action in actions:
            args = {}
            
            for k, v in action.arguments().items():
                if k != "caller_context":
                    args[k] = v
                    
            actions_list.append(f"{action.name()}({', '.join([f.format(k , v) for k, v in args.items()])}) // {action.description()}")
            
        
        actions_str = linesep.join(actions_list)

        response = await self._language_model.query(query, f"{self._system_prompt}{linesep}{actions_str}")
        return await self._parse_response(response)

    async def _parse_response(self, json_text: str, retry_count: int = 3) -> Dict[str, Any]:
        """
        Parses the json and returns a string
        """
        try:
            json_text = json_text.strip()
            if "```json" in json_text:
                return orjson.loads(json_text.split("```json")[1].split("```")[0])
            if json_text.startswith("```"):
                json_text = json_text.lstrip("```")
            if json_text.endswith("```"):
                json_text = json_text.rstrip("```")
            return orjson.loads(json_text)
        except (orjson.JSONDecodeError, Exception):
            if retry_count > 0:
                retry_count -= 1
                return await self._parse_response(await self._fix_json(json_text), retry_count)
            return {"action": "error", "args": {"error": "Response is not a valid JSON document"}}

    async def _fix_json(self, json_text: str) -> str:
        return await self._language_model.query(
            json_text, "You have received the following json, fix it if it has any errors and respond with fixed JSON, if it is already correct, simply respond the same.")

    def info(self) -> EntityInfo:
        return EntityInfo(
            name=self.__class__.__name__,
            version="0.1.0",
            params={
                "system_prompt": self._system_prompt,
                "response_template": self._response_template,
                "wait_timeout_s": self._wait_timeout_s,
                "actions": [
                    {
                        "name": action.name(),
                        "args": action.arguments(),
                        "description": action.description()
                    } for action in self.action_space.all_actions()
                ]
            }

        )
