from typing import Any, Dict, List
from os import linesep

from agentopy import WithStateMixin, WithActionSpaceMixin, Action, ActionResult, IEnvironmentComponent, EntityInfo


class TodoList(WithStateMixin, WithActionSpaceMixin, IEnvironmentComponent):
    """
    Implements a todo list component
    """

    def __init__(self) -> None:
        super().__init__()
        self._list: List[str] = []

        self.action_space.register_actions([
            Action(
                "add_item_to_todo_list", "use this action to add an item to the todo list", self.add),
            Action(
                "remove_item_from_todo_list", "use this action to remove an item from the todo list", self.remove),
            Action(
                "clean_todo_list", "Clears the todo list", self.clear),
            Action("check_todo_list", "use this action to list all items in the todo list", self.get_all)])

    async def add(self, item: str) -> ActionResult:
        """
        Adds the specified item to the todo list
        """
        self._list.append(item)
        return ActionResult(value="OK", success=True)

    async def remove(self, item: str) -> ActionResult:
        """
        Removes the specified item from the todo list
        """
        try:
            self._list.remove(item)
            return ActionResult(value="OK", success=True)
        except ValueError:
            return ActionResult(value="No such item in the todo list", success=False)

    async def clear(self) -> ActionResult:
        """
        Clears the todo list
        """
        self._list.clear()
        return ActionResult(value="OK", success=True)

    async def get_all(self) -> ActionResult:
        """
        Returns all items in the todo list
        """
        return ActionResult(value=linesep.join(self._list), success=True)

    async def on_tick(self) -> None:
        self._state.set_item(
            "status", {"num_new": len(self._list)})

    def info(self) -> EntityInfo:
        return EntityInfo(
            name="TodoList",
            version="0.1.0",
            params={}
        )
