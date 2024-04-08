from typing import List
from os import linesep
import asyncio as aio
from agentopy import WithActionSpaceMixin, Action, ActionResult, IEnvironmentComponent, EntityInfo, State, IState


class TodoList(WithActionSpaceMixin, IEnvironmentComponent):
    """
    Implements a todo list component
    """

    def __init__(self) -> None:
        super().__init__()
        self._list: List[str] = []

        self.action_space.register_actions([
            Action(
                "add_item_to_todo_list", "use this action to add an item to the todo list", self.add, self.info()),
            Action(
                "remove_item_from_todo_list", "use this action to remove an item from the todo list", self.remove, self.info()),
            Action(
                "clean_todo_list", "Clears the todo list", self.clear, self.info()),
            Action("check_todo_list", "use this action to list all items in the todo list", self.get_all, self.info())
        ])

    async def add(self, *, item: str, caller_context: IState) -> ActionResult:
        """
        Adds the specified item to the todo list
        """
        self._list.append(item)
        return ActionResult(value="OK", success=True)

    async def remove(self, *, item: str, caller_context: IState) -> ActionResult:
        """
        Removes the specified item from the todo list
        """
        try:
            self._list.remove(item)
            return ActionResult(value="OK", success=True)
        except ValueError:
            return ActionResult(value="No such item in the todo list", success=False)

    async def clear(self, *, caller_context: IState) -> ActionResult:
        """
        Clears the todo list
        """
        self._list.clear()
        return ActionResult(value="OK", success=True)

    async def get_all(self, *, caller_context: IState) -> ActionResult:
        """
        Returns all items in the todo list
        """
        return ActionResult(value=linesep.join(self._list), success=True)

    async def tick(self) -> None:
        await aio.sleep(10)
        
    async def observe(self, caller_context: IState) -> State:
        state = State()
        state.set_item("status", {"num_new": len(self._list)})
        return state

    def info(self) -> EntityInfo:
        return EntityInfo(
            name=self.__class__.__name__,
            version="0.1.0",
            params={}
        )
