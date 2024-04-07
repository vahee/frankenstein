from agentopy import ActionResult, IAgentComponent, WithActionSpaceMixin, Action, EntityInfo, IAgent, IState
from frankenstein.lib.language.protocols import ILanguageModel


class Creativity(WithActionSpaceMixin, IAgentComponent):
    """Implements a creativity component"""

    def __init__(self, language_model: ILanguageModel):
        """Initializes the creativity component"""
        super().__init__()
        self._language_model: ILanguageModel = language_model

        self.action_space.register_actions(
            [
                Action(
                    "generate_text", "use this action to return textual content by improvising, for example articles, blog posts, poems, essays, etc.", self.text, self.info()),
                Action(
                    "brainstorm", "use this action to generate ideas for a specific objective", self.ideas, self.info())
            ])

    async def text(self, *, objective: str, context: str | None = None, caller_context: IState) -> ActionResult:
        """Creates text content"""
        context = f"""
        Act as a writer. You write any kind of text, given an objective.
        {f"Here is some additional context: {context}" if context is not None else ""}
        """
        lm_response = await self._language_model.query(f"The topic is {objective}", context)
        return ActionResult(value=lm_response, success=True)

    async def ideas(self, *, objective: str, context: str | None = None, caller_context: IState) -> ActionResult:
        """Creates ideas for a topic"""
        context = f"""
        Act as a professional ideator. You generate ideas for a given objective.
        {f"Here is some additional context: {context}" if context is not None else ""}
        """
        lm_response = await self._language_model.query(f"The objective is \"{objective}\"", context)
        return ActionResult(value=lm_response, success=True)

    async def on_agent_heartbeat(self, agent: IAgent) -> None:
        agent.state.set_item(f"agent/components/{self.info().name}/status", "Creativity is active.")
    
    async def tick(self) -> None:
        ...

    def info(self) -> EntityInfo:
        return EntityInfo(
            name=self.__class__.__name__,
            version="0.1.0",
            params={}
        )
