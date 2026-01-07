from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart
from a2a.utils import get_message_text

from .player import Player


class Agent:
    def __init__(self):
        self.player = Player()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Implement your agent logic here.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        input_text = get_message_text(message)
        skip_response = (message.metadata or {}).get("skip_response", False)
        print(">>> " + input_text)
        statement = await self.player.handle(input_text, skip_response=skip_response)
        if not skip_response:
            print("<<< " + statement)

        print("")

        # Replace this example code with your agent logic

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=statement))],
            name="Response",
        )
