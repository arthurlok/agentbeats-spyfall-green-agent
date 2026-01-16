import logging
import random
from typing import Any
from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from .messenger import Messenger
from .game_env import SpyfallEnv, all_locations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("green_agent")

class EvalRequest(BaseModel):
    """
    Request format sent by the AgentBeats platform to green agents.
    
    Attributes:
        participants: Mapping of player names to their A2A agent endpoint URLs
        config: Game configuration containing location and num_rounds
    """
    participants: dict[str, HttpUrl] # name -> agent URL
    config: dict[str, Any]


class Agent:
    """
    Green agent that orchestrates a Spyfall game.
    
    This agent receives a request with participant URLs and game configuration,
    then uses the SpyfallEnv to coordinate the game between multiple purple agents.
    """
    # Required configuration keys for a Spyfall game
    required_config_keys: list[str] = ["location", "num_rounds"]

    def __init__(self):
        self.messenger = Messenger()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """
        Validate that the request has all required configuration.

        Args:
            request: The evaluation request to validate

        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        num_players = len(request.participants)
        if num_players < 3:
            return False, f"Minimum 3 players required, got {num_players}"
        if num_players > 8:
            return False, f"Maximum 8 players allowed, got {num_players}"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Run a Spyfall game with the provided participants and configuration.

        Args:
            message: The incoming A2A message containing participants and config
            updater: A2A task updater for reporting progress and results
        """
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        # Resolve location - if "random", pick from all_locations
        location = request.config["location"]
        if location.lower() == "random":
            location = random.choice(all_locations)

        # Update status with game start information
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting Spyfall game.\nParticipants: {list(request.participants.keys())}\nLocation: {location}\nRounds: {request.config['num_rounds']}")
        )

        # Run the Spyfall game
        game_result = await self.run_single_game(
            participants=request.participants,
            location=location,
            num_rounds=request.config["num_rounds"],
            updater=updater
        )

        # Format result message
        result_text = self._format_game_result(game_result)

        # Add artifact with game results
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=result_text)),
                Part(root=DataPart(data=game_result))
            ],
            name="Game Result",
        )

    async def run_single_game(self, participants: dict[str, HttpUrl], location: str, num_rounds: int, updater: TaskUpdater) -> dict:
        """
        Run a single game of Spyfall and return the results.
        
        Args:
            participants: Dictionary mapping participant names to their agent URLs
            location: The secret location for the game
            num_rounds: Maximum number of rounds to play
            updater: Task updater for reporting progress
            
        Returns:
            Dictionary containing game results
        """
        # Create game environment with the provided configuration
        game_env = SpyfallEnv(participants=participants, location=location, max_rounds=num_rounds)
        
        # Randomly assign one participant as spy, others as non-spies
        assigned_roles = game_env.assign_roles()
        spy_name = [name for name, role in assigned_roles.items() if role == "spy"][0]
        
        logger.info(f"Game started - Spy: {spy_name}, Location: {location}, Max Rounds: {num_rounds}")
        
        # Run the game loop (initialization, action rounds, voting/end condition)
        game_result = await game_env.play_game(assigned_roles, location)
        
        spy = next((p for p in game_result['players'] if p['role'] == 'spy'), None)
        winner_role = "spy" if spy and spy['won'] else "non-spies"
        logger.info(f"Game ended - Winner: {winner_role}")
        
        return game_result

    def _format_game_result(self, game_result: dict) -> str:
        """
        Format the game result into a readable string.

        Args:
            game_result: The game result dictionary

        Returns:
            Formatted result string for human readability
        """
        # Find spy and winners from players list
        spy = next((p for p in game_result['players'] if p['role'] == 'spy'), None)
        winners = [p['name'] for p in game_result['players'] if p['won']]
        winner_role = "spy" if spy and spy['won'] else "non-spies"

        result_text = f"""
Spyfall Game Results
===================

Winner: {winner_role} ({', '.join(winners)})
End Method: {game_result['end_method']}

Spy: {spy['name'] if spy else 'Unknown'}
Result: {game_result['result']}
"""
        if game_result['end_method'] == 'vote':
            result_text += f"""
Voting Results:
- Players voted for: {game_result['voted_as_spy']}
- Vote breakdown: {game_result['votes']}
"""

        return result_text
