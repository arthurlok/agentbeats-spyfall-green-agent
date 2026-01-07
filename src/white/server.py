import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from .executor import Executor


def start(host, port, card_url):
    skill = AgentSkill(
        id="play_spyfall",
        name="Play Spyfall",
        description="Play Spyfall.",
        tags=[],
        examples=[],
    )

    agent_card = AgentCard(
        name="Spyfall Player",
        description="Player for Spyfall game.",
        url=card_url or f"http://{host}:{port}/",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=host, port=port)
