from langchain_core.messages import BaseMessage


class HistoryManager:
    """Format the last configured number of conversational turns."""

    def __init__(self, turns: int) -> None:
        self.turns = turns

    def trim(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        return messages[-self.turns * 2 :]

    def format(self, messages: list[object]) -> str:
        # The latest user message is supplied separately to the router/rewriter.
        history = messages[:-1][-self.turns * 2 :]
        return "\n".join(f"{getattr(message, 'type', 'message')}: {getattr(message, 'content', message)}" for message in history)