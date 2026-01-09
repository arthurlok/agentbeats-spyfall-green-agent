import json
import logging
import random
import asyncio
from typing import Literal
from pydantic import BaseModel, Field, HttpUrl, ValidationError

from .messenger import Messenger

logger = logging.getLogger("spyfall_env")

all_locations = [
    "Airplane",
    "Bank",
    "Beach",
    "Broadway Theater",
    "Casino",
    "Cathedral",
    "Circus Tent",
    "Corporate Party",
    "Crusader Army",
    "Day Spa",
    "Embassy",
    "Hospital",
    "Hotel",
    "Military Base",
    "Movie Studio",
    "Ocean Liner",
    "Passenger Train",
    "Pirate Ship",
    "Polar Station",
    "Police Station",
    "Restaurant",
    "School",
    "Service Station",
    "Space Station",
    "Submarine",
    "Supermarket",
    "University",
]

# Final results schema

# {
#     "winner": "spy" or "non-spies",
#     "winner_role": "spy" or "non-spy",
#     "participants": [...],
#     "end_method": "vote" or "spy_guess",
#     "spy": <spy_name>,
#     "voted_as_spy": <name or None>,
#     "votes": {...} or None,
#     "result": <message>
# }

# Action models for structured JSON responses
class AskQuestionAction(BaseModel):
    """Spy or non-spy asking a question to another player."""
    action: Literal["ask_question"] = Field(description="The action type")
    target: str = Field(description="Name of the participant to ask")
    question: str = Field(description="The question to ask")


class GuessLocationAction(BaseModel):
    """Spy guessing the location."""
    action: Literal["guess_location"] = Field(description="The action type")
    location_guess: str = Field(description="The spy's guess for the location")


NonSpyAction = AskQuestionAction  # Non-spies can only ask questions


def get_action_schema(is_spy: bool) -> str:
    """Get JSON schema for valid actions."""
    if is_spy:
        # Spy can ask question or guess location
        ask_schema = json.dumps(AskQuestionAction.model_json_schema(), indent=2)
        guess_schema = json.dumps(GuessLocationAction.model_json_schema(), indent=2)
        return f"""Spy Actions (choose one):

Ask Question:
{ask_schema}

Guess Location:
{guess_schema}"""
    else:
        # Non-spy can only ask questions
        schema = json.dumps(NonSpyAction.model_json_schema(), indent=2)
        return f"""Non-Spy Action:

{schema}"""


def extract_json_from_response(response: str) -> str:
    """
    Extract JSON from a response that may contain markdown code fences.

    LLMs often return JSON wrapped in markdown like:
    ```json
    {"key": "value"}
    ```

    This function strips the code fences to get the raw JSON.
    """
    text = response.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        # Find the end of the first line (which may be ```json or just ```)
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]

        # Remove closing fence
        if text.endswith("```"):
            text = text[:-3]

    return text.strip()


def parse_action(response: str, is_spy: bool) -> dict | None:
    """
    Parse and validate JSON action from agent response.

    Agents must respond with valid JSON during action phases. This function extracts
    and validates the action according to the agent's role (spy vs non-spy).

    Args:
        response: The raw response text from the agent
        is_spy: Whether the responding agent is a spy or non-spy

    Returns:
        Validated action dictionary, or None if parsing/validation fails

    Note:
        - Spy can perform: ask_question or guess_location
        - Non-spy can only perform: ask_question
        - Invalid actions are logged as warnings and return None
    """
    try:
        # Extract JSON from response (handles markdown code fences)
        json_str = extract_json_from_response(response)
        action_dict = json.loads(json_str)
        
        if is_spy:
            # Try to parse as either ask or guess action
            if action_dict.get("action") == "ask_question":
                AskQuestionAction.model_validate(action_dict)
            elif action_dict.get("action") == "guess_location":
                GuessLocationAction.model_validate(action_dict)
            else:
                logger.warning(f"Invalid spy action: {action_dict}")
                return None
        else:
            # Non-spy can only ask questions
            NonSpyAction.model_validate(action_dict)
        
        return action_dict
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning(f"Failed to parse action: {e}")
        return None

class SpyfallEnv:
    """
    Orchestrates a game of Spyfall between multiple agents.
    
    Game Flow:
    1. Initialize: Send role information and game rules to all agents
    2. Action Rounds: Each round, players take turns asking questions or making guesses
    3. End Game: After max_rounds or when spy guesses, conduct voting to determine winner
    
    The environment maintains the game state and coordinates communication between
    the green agent (orchestrator) and white agents (players) via the A2A protocol.
    """

    def __init__(self, participants: dict[str, HttpUrl], location: str, max_rounds: int = 5):
        """
        Initialize the Spyfall game environment.
        
        Args:
            participants: Dict mapping player names to their A2A endpoint URLs
            location: The secret location for this game
            max_rounds: Maximum number of rounds before voting (default: 5)
        """
        self.participants = participants # name -> agent URL
        self.location = location
        self.round = 0
        self.max_rounds = max_rounds
        self.assigned_roles = {} # similar structure to participants, but instead of storing URLs, store roles
        self.messenger = Messenger()
        self.game_over = False
        self.spy_win = False 
    
    def assign_roles(self) -> dict[str, str]:
        """Randomly assign one participant as spy, others as non-spies."""
        assigned_roles = {}
        spy_participant = random.choice(list(self.participants.keys()))
        for name in self.participants:
            assigned_roles[name] = "spy" if name == spy_participant else "non-spy"
        self.assigned_roles = assigned_roles
        return assigned_roles

    def _get_spy(self, assigned_roles: dict[str, str]) -> str:
        """Get the name of the spy from assigned roles."""
        return [name for name, role in assigned_roles.items() if role == "spy"][0]

    def _get_non_spies(self, assigned_roles: dict[str, str]) -> list[str]:
        """Get list of non-spy player names from assigned roles."""
        return [name for name, role in assigned_roles.items() if role == "non-spy"]

    def _get_other_players(self, assigned_roles: dict[str, str], exclude_name: str) -> list[str]:
        """Get list of all players except the excluded one."""
        return [name for name in self.participants if name != exclude_name]

    def _build_spy_init_prompt(self, player_name: str, assigned_roles: dict[str, str]) -> str:
        """Build the initialization prompt for the spy."""
        other_players = self._get_other_players(assigned_roles, player_name)
        return f"""You are playing a game of Spyfall. Your role is spy. The location is unknown to you but known to the remaining players (non-spies).
                        Each round, all players will take a turn each asking and answering questions about the location. 
                        You will not necessarily get to ask or answer a question every turn, but the conversation will be broadcasted to you and all players.
                        Non-spies will attempt to identify the spy while not revealing too much about the location to the spy.
                        As the spy, you must try to identify the location through the conversation and also ask/answer questions without raising suspicion.

                        These are the names of the other players: {', '.join(other_players)}.

                        These are all possible locations for the game: {', '.join(all_locations)}.

                        If you feel that you have enough information to guess the location, you may do so on your turn instead of asking a question.
                        If you guess the location correctly, you win the game immediately. Otherwise, you lose immediately.

                        You will play a total of {self.max_rounds} rounds. At the end of the game, all non-spies will blindly vote on who they believe the spy is. 
                        If the non-spy majority correctly identifies the spy, they win; otherwise, the spy wins.
                        """

    def _build_non_spy_init_prompt(self, player_name: str, location: str, assigned_roles: dict[str, str]) -> str:
        """Build the initialization prompt for a non-spy."""
        other_players = self._get_other_players(assigned_roles, player_name)
        return f"""You are playing a game of Spyfall. Your role is non-spy. The location is {location}.
                        Each round, all players will take a turn each asking and answering questions about the location.
                        The spy does not know the location and will try to blend in while identifying the location through the conversation. 
                        As a non-spy, you must try to identify who the spy is through questioning and observation without obviously revealing the location. 
                        You will not necessarily get to ask or answer a question every turn, but the conversation will be broadcasted to you and all players.
                        
                        These are the names of the other players: {', '.join(other_players)}.

                        These are all possible locations for the game: {', '.join(all_locations)}.
                        
                        Ask questions strategically to catch the spy, and answer questions honestly to help other non-spies identify the spy.

                        If a spy chooses to guess the location on their turn and guesses correctly, they win the game immediately. Otherwise, they lose immediately.

                        You will play a total of {self.max_rounds} rounds. At the end of the game, all non-spies will blindly vote on who they believe the spy is. 
                        If the non-spy majority correctly identifies the spy, they win; otherwise, the spy wins.
                        """

    def _build_spy_action_prompt(self, assigned_roles: dict[str, str]) -> str:
        """Build the action prompt for the spy."""
        action_schema = get_action_schema(is_spy=True)
        return f"""It is now your turn to take an action. You must choose ONE of the following actions:

{action_schema}

Examples of valid responses:

Ask a question:
{json.dumps({"action": "ask_question", "target": "Alice", "question": "Is this location outdoors?"}, indent=2)}

Guess the location:
{json.dumps({"action": "guess_location", "location_guess": "Beach"}, indent=2)}

Respond ONLY with valid JSON matching one of the schemas above. Do not include any other text.
                        """

    def _build_non_spy_action_prompt(self) -> str:
        """Build the action prompt for a non-spy."""
        action_schema = get_action_schema(is_spy=False)
        return f"""It is now your turn to ask a question to help identify the spy.

{action_schema}

Example of a valid response:
{json.dumps({"action": "ask_question", "target": "Bob", "question": "Is this location a public place?"}, indent=2)}

Respond ONLY with valid JSON matching the schema above. Do not include any other text.
                        """

    def _build_game_result(self, winner: str, spy_player: str, participants: list[str], 
                           end_method: str, result_message: str, voted_as_spy: str = None, 
                           votes: dict = None) -> dict:
        """Build the standardized game result dictionary."""
        return {
            "winner": winner,
            "winner_role": "spy" if winner == "spy" else "non-spy",
            "participants": participants,
            "end_method": end_method,
            "spy": spy_player,
            "voted_as_spy": voted_as_spy,
            "votes": votes,
            "result": result_message
        }

    async def answer_turn(self, target: str, asker: str, question: str) -> str:
        """
        Send a question to a target player and collect their response.
        
        Args:
            target: Name of the player being asked
            asker: Name of the player asking
            question: The question being asked
            
        Returns:
            The target player's response
        """
        prompt = f"""A player named {asker} asks you: "{question}"

You must answer this question in natural language (NOT JSON). 

Give a conversational response that gives you the best chance to win the game based on your role."""
        response = await self.messenger.talk_to_agent(
            prompt, str(self.participants[target]), new_conversation=False
        )
        logger.info(f"{target} answered {asker}'s question: {response}")
        return response

    async def _broadcast(self, message: str, skip_response: bool = True) -> None:
        """
        Send a message to all players concurrently.
        
        Args:
            message: The message to broadcast to all players
            skip_response: If True, don't process responses from players (default: True)
        """
        tasks = []
        for player_name in self.participants:
            # Send message to each player with metadata
            tasks.append(
                self.messenger.talk_to_agent(
                    message, 
                    str(self.participants[player_name]), 
                    new_conversation=False,
                    metadata={"skip_response": skip_response}
                )
            )
        # Execute all sends concurrently, ignore responses
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _collect_votes(self, assigned_roles: dict[str, str]) -> dict[str, int]:
        """
        Collect votes from all non-spy players on who they believe the spy is.
        
        Args:
            assigned_roles: Mapping of player names to roles
            
        Returns:
            Dictionary with vote counts for each player
        """
        non_spy_players = self._get_non_spies(assigned_roles)
        spy_player = self._get_spy(assigned_roles)
        other_players = self._get_other_players(assigned_roles, spy_player)
        
        votes = {player: 0 for player in self.participants}
        
        for voter_name in non_spy_players:
            prompt = f"""The game has ended. Now it is time to vote on who you think the spy is.
            
You must vote for one of the other players. Here are your choices:
{', '.join(other_players)}

Respond with ONLY the name of the player you believe is the spy. Do not include any other text.
            """
            response = await self.messenger.talk_to_agent(
                prompt, str(self.participants[voter_name]), new_conversation=False
            )
            
            # Parse the vote - look for player names in the response
            vote = response.strip()
            if vote in self.participants:
                votes[vote] += 1
                logger.info(f"{voter_name} voted for {vote}")
            else:
                logger.warning(f"Invalid vote from {voter_name}: {vote}")
        
        return votes

    async def _end_game(self, assigned_roles: dict[str, str]) -> dict:
        """
        End the game and determine the winner based on voting.
        
        Args:
            assigned_roles: Mapping of player names to roles
            
        Returns:
            Dictionary with game results
        """
        spy_player = self._get_spy(assigned_roles)
        
        # Collect votes from non-spies
        votes = await self._collect_votes(assigned_roles)
        
        # Find who got the most votes
        max_votes = max(votes.values()) if votes.values() else 0
        players_with_max_votes = [name for name, count in votes.items() if count == max_votes]
        
        # In case of a tie, pick the first one (or could implement tiebreaker)
        voted_as_spy = players_with_max_votes[0] if players_with_max_votes else None
        
        logger.info(f"Vote results: {votes}")
        logger.info(f"Majority voted for: {voted_as_spy}")
        
        # Determine winner
        if voted_as_spy == spy_player:
            # Non-spies correctly identified the spy
            winner = "non-spies"
            self.spy_win = False
            result_message = f"The non-spies correctly identified {spy_player} as the spy! Non-spies win!"
            logger.info(result_message)
        else:
            # Non-spies failed to identify the spy
            winner = "spy"
            self.spy_win = True
            result_message = f"The non-spies voted for {voted_as_spy}, but {spy_player} was actually the spy! Spy wins!"
            logger.info(result_message)
        
        # Broadcast the results
        await self._broadcast(result_message)
        
        return self._build_game_result(
            winner=winner,
            spy_player=spy_player,
            participants=list(self.participants.keys()),
            end_method="vote",
            result_message=result_message,
            voted_as_spy=voted_as_spy,
            votes=votes
        )

    
    async def play_game(self, assigned_roles: dict[str, str], location: str) -> dict[str, str]:
        while self.round < self.max_rounds and not self.game_over:
            if self.round == 0:
                # Initial round - send role information to all players
                await self._send_initial_prompts(assigned_roles, location)
                self.round += 1
            else:
                # Subsequent rounds - players take turns with actions
                await self._run_action_round(assigned_roles)
                self.round += 1

        # Determine game result
        spy_player = self._get_spy(assigned_roles)
        
        if not self.game_over:
            # Game ended naturally (max rounds reached) - conduct voting
            game_results = await self._end_game(assigned_roles)
            return game_results
        else:
            # Game ended early due to spy guess
            result_message = f"The spy {spy_player} guessed the location: {'correct' if self.spy_win else 'incorrect'}!"
            return self._build_game_result(
                winner="spy" if self.spy_win else "non-spies",
                spy_player=spy_player,
                participants=list(self.participants.keys()),
                end_method="spy_guess",
                result_message=result_message
            )

    async def _send_initial_prompts(self, assigned_roles: dict[str, str], location: str) -> None:
        """Send initial game information to all players."""
        for name in assigned_roles:
            if assigned_roles[name] == "spy":
                prompt = self._build_spy_init_prompt(name, assigned_roles)
            else:
                prompt = self._build_non_spy_init_prompt(name, location, assigned_roles)
            
            await self.messenger.talk_to_agent(
                prompt, str(self.participants[name]), new_conversation=True
            )

    async def _run_action_round(self, assigned_roles: dict[str, str]) -> None:
        """Run a round where each player takes an action."""
        for name in assigned_roles:
            # Stop if game ends early (e.g., spy makes a guess)
            if self.game_over:
                break
            
            if assigned_roles[name] == "spy":
                await self._process_spy_action(name, assigned_roles)
            else:
                await self._process_non_spy_action(name, assigned_roles)

    async def _process_spy_action(self, spy_name: str, assigned_roles: dict[str, str]) -> None:
        """Process the spy's turn - get their action and handle it."""
        prompt = self._build_spy_action_prompt(assigned_roles)
        response = await self.messenger.talk_to_agent(
            prompt, str(self.participants[spy_name]), new_conversation=False
        )
        
        action = parse_action(response, is_spy=True)
        if action:
            logger.info(f"Spy {spy_name} action: {action}")
            await self._handle_action(spy_name, action, assigned_roles)
        else:
            logger.warning(f"Failed to parse spy action from {spy_name}")

    async def _process_non_spy_action(self, non_spy_name: str, assigned_roles: dict[str, str]) -> None:
        """Process a non-spy's turn - get their action and handle it."""
        prompt = self._build_non_spy_action_prompt()
        response = await self.messenger.talk_to_agent(
            prompt, str(self.participants[non_spy_name]), new_conversation=False
        )
        
        action = parse_action(response, is_spy=False)
        if action:
            logger.info(f"Non-spy {non_spy_name} action: {action}")
            await self._handle_action(non_spy_name, action, assigned_roles)
        else:
            logger.warning(f"Failed to parse non-spy action from {non_spy_name}")
    
    async def _handle_action(self, actor: str, action: dict, assigned_roles: dict[str, str]) -> None:
        """
        Handle an action taken by a player.
        
        Args:
            actor: Name of the player taking the action
            action: The parsed action dictionary
            assigned_roles: Mapping of player names to roles
        """
        if action.get("action") == "ask_question":
            target = action.get("target")
            question = action.get("question")
            
            if target not in self.participants:
                logger.warning(f"Invalid target {target} for question from {actor}")
                return
            
            logger.info(f"{actor} asks {target}: {question}")
            # Get the answer from the target player
            answer = await self.answer_turn(target, actor, question)
            
            # Broadcast the question and answer to all players
            broadcast_message = f"""Question asked during the game:
{actor}: "{question}" (asked to {target})
{target}: "{answer}"
            """
            await self._broadcast(broadcast_message)

        # Handle spy choosing to guess the location
        
        elif action.get("action") == "guess_location":
            location_guess = action.get("location_guess")
            logger.info(f"{actor} guesses: {location_guess}")
            
            if location_guess.lower() == self.location.lower():
                logger.info(f"Spy {actor} guessed correctly!")
                self.game_over = True
                self.spy_win = True
                # Broadcast the correct guess
                broadcast_message = f"{actor} (the spy) guessed the location correctly: {location_guess}! The spy wins!"
                await self._broadcast(broadcast_message)
            else:
                logger.info(f"Spy {actor} guessed incorrectly. The location was {self.location}")
                self.game_over = True
                self.spy_win = False
