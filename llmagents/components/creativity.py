from typing import Dict, Any
from agentopy import ActionResult, IEnvironmentComponent, WithActionSpaceMixin, WithStateMixin, Action, EntityInfo
from llmagents.language.protocols import ILanguageModel


class Creativity(WithStateMixin, WithActionSpaceMixin, IEnvironmentComponent):
    """Implements a creativity environment component"""

    def __init__(self, language_model: ILanguageModel):
        """Initializes the creativity environment component"""
        super().__init__()
        self._language_model: ILanguageModel = language_model

        self.action_space.register_actions(
            [
                Action(
                    "generate_text", "use this action to return textual content by improvising, for example articles, blog posts, poems, essays, etc.", self.text),
                Action(
                    "brainstorm", "use this action to generate ideas for a specific objective", self.ideas)
            ])

    async def text(self, objective: str, context: str | None = None) -> ActionResult:
        """Creates text content"""
        context = f"""
        Act as a writer. You write any kind of text, given an objective.
        {f"Here is some additional context: {context}" if context is not None else ""}
        """
        lm_response = await self._language_model.query(f"The topic is {objective}", context)
        return ActionResult(value=lm_response, success=True)

    async def ideas(self, objective: str, context: str | None = None) -> ActionResult:
        """Creates ideas for a topic"""
        context = f"""
        Act as a professional ideator. You generate ideas for a given objective.
        {f"Here is some additional context: {context}" if context is not None else ""}
        """
        lm_response = await self._language_model.query(f"The objective is \"{objective}\"", context)
        return ActionResult(value=lm_response, success=True)

    async def on_tick(self) -> None:
        self._state.set_item("status", "Creativity is active.")

    def info(self) -> EntityInfo:
        return EntityInfo(
            name="Creativity",
            version="0.1.0",
            params={}
        )
