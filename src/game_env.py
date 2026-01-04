import json
import logging
import random
import asyncio
from typing import Literal, Union
from pydantic import BaseModel, Field, HttpUrl, ValidationError

from messenger import Messenger

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


class SpyAction(BaseModel):
    """Action for a spy player."""
    __root__: Union[AskQuestionAction, GuessLocationAction] = Field(discriminator="action")


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


def parse_action(response: str, is_spy: bool) -> dict | None:
    """Parse and validate JSON action from agent response."""
    try:
        # Try to extract JSON from response
        action_dict = json.loads(response)
        
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

    def __init__(self, participants: dict[str, HttpUrl], location: str, max_rounds: int = 5):
        self.participants = participants # name -> agent URL
        self.location = location
        self.round = 0
        self.max_rounds = max_rounds
        self.assigned_roles = {} # similar structure to participants, but instead of storing URLs, store roles
        self.messenger = Messenger()
        self.game_over = False
        self.spy_win = False 
    
    def assign_roles(self) -> dict[str, str]:
        # Logic to assign roles to players
        assigned_roles = {}
        ## Randomly assign one participant as spy
        spy_participant = random.choice(list(self.participants.keys()))
        for name in self.participants:
            if name == spy_participant:
                assigned_roles[name] = "spy"
            else:
                assigned_roles[name] = "non-spy"
        self.assigned_roles = assigned_roles
        return assigned_roles

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
        Please provide your answer. 
        """
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
        non_spy_players = [name for name, role in assigned_roles.items() if role == "non-spy"]
        spy_player = [name for name, role in assigned_roles.items() if role == "spy"][0]
        
        votes = {player: 0 for player in self.participants}
        
        # Get other player names for the voting prompt
        other_players = [name for name in self.participants if name != spy_player]
        
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
        spy_player = [name for name, role in assigned_roles.items() if role == "spy"][0]
        
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
            self.spy_win = False
            result_message = f"The non-spies correctly identified {spy_player} as the spy! Non-spies win!"
            logger.info(result_message)
        else:
            # Non-spies failed to identify the spy
            self.spy_win = True
            result_message = f"The non-spies voted for {voted_as_spy}, but {spy_player} was actually the spy! Spy wins!"
            logger.info(result_message)
        
        # Broadcast the results
        await self._broadcast(result_message)
        
        return {
            "spy": spy_player,
            "voted_as_spy": voted_as_spy,
            "votes": votes,
            "spy_win": self.spy_win,
            "result": result_message
        }

    
    async def play_game(self, assigned_roles: dict[str, str], location: str) -> dict[str, str]:
        while self.round < self.max_rounds and not self.game_over:
            if self.round == 0:
                # Initial round logic
                for name in assigned_roles:
                    if assigned_roles[name] == "spy":
                        prompt = f"""You are playing a game of Spyfall. Your role is spy. The location is unknown to you but known to the remaining players (non-spies).
                        Each round, all players will take turns asking and answering questions about the location. 
                        You will not necessarily get to ask or answer a question every round, but the conversation will be broadcasted to you and all players.
                        Non-spies will attempt to identify the spy while not revealing too much about the location to the spy.
                        As the spy, you must try to identify the location through the conversation and also ask/answer questions without revealing your identity.

                        These are the names of the other players: {', '.join([n for n in assigned_roles if n != name])}.

                        These are all possible locations for the game: {', '.join(all_locations)}.

                        If you feel that you have enough information to guess the location, you may do so on your turn instead of asking a question.
                        If you guess the location correctly, you win the game immediately. Otherwise, you lose immediately.

                        You will play a total of {self.max_rounds} rounds. At the end of the game, all non-spies will blindly vote on who they believe the spy is. 
                        If the non-spy majority correctly identifies the spy, they win; otherwise, the spy wins.
                        """
                        response = await self.messenger.talk_to_agent(prompt, str(self.participants[name]), new_conversation=True)
                    else:
                        prompt = f"""You are playing a game of Spyfall. Your role is non-spy. The location is {location}.
                        Each round, all non-spy players will take turns asking and answering questions about the location to attempt to identify the spy, who is the only player that does not know the location.
                        As a non-spy, you must try to identify who the spy is through the conversation while answering questions without revealing the location. 
                        You will not necessarily get to ask or answer a question every round, but the conversation will be broadcasted to you and all players.
                        

                        These are the names of the other players: {', '.join([n for n in assigned_roles if n != name])}.

                        These are all possible locations for the game: {', '.join(all_locations)}.
                        
                        Ask questions strategically to catch the spy, and answer questions honestly to help other non-spies identify the spy.

                        If a spy chooses to guess the location on their turn and guesses correctly, they win the game immediately. Otherwise, they lose immediately.

                        You will play a total of {self.max_rounds} rounds. At the end of the game, all non-spies will blindly vote on who they believe the spy is. 
                        If the non-spy majority correctly identifies the spy, they win; otherwise, the spy wins.
                        """
                        response = await self.messenger.talk_to_agent(prompt, str(self.participants[name]), new_conversation=True)

                self.round += 1
            # Further rounds logic
            else:
                for name in assigned_roles:
                    # Check if game is over (e.g., spy made a guess)
                    if self.game_over:
                        break
                    
                    if assigned_roles[name] == "spy":
                        action_schema = get_action_schema(is_spy=True)
                        prompt = f"""It is now your turn to take an action. You must choose ONE of the following actions:

{action_schema}

Examples of valid responses:

Ask a question:
{json.dumps({{"action": "ask_question", "target": "Alice", "question": "Is this location outdoors?"}}, indent=2)}

Guess the location:
{json.dumps({{"action": "guess_location", "location_guess": "Beach"}}, indent=2)}

Respond ONLY with valid JSON matching one of the schemas above. Do not include any other text.
                        """
                        response = await self.messenger.talk_to_agent(prompt, str(self.participants[name]), new_conversation=False)
                        action = parse_action(response, is_spy=True)
                        if action:
                            logger.info(f"Spy {name} action: {action}")
                            # Handle the action
                            await self._handle_action(name, action, assigned_roles)
                        else:
                            logger.warning(f"Failed to parse spy action from {name}")
                    else:
                        action_schema = get_action_schema(is_spy=False)
                        prompt = f"""It is now your turn to ask a question to help identify the spy.

{action_schema}

Example of a valid response:
{json.dumps({"action": "ask_question", "target": "Bob", "question": "Is this location a public place?"}, indent=2)}

Respond ONLY with valid JSON matching the schema above. Do not include any other text.
                        """
                        response = await self.messenger.talk_to_agent(prompt, str(self.participants[name]), new_conversation=False)
                        action = parse_action(response, is_spy=False)
                        if action:
                            logger.info(f"Non-spy {name} action: {action}")
                            # Handle the action
                            await self._handle_action(name, action, assigned_roles)
                        else:
                            logger.warning(f"Failed to parse non-spy action from {name}")

                self.round += 1

        # Game is over - conduct voting if it ended due to max rounds
        if not self.game_over:
            game_results = await self._end_game(assigned_roles)
            return game_results
        else:
            # Game ended early due to spy guess
            return {
                "game_ended_early": True,
                "spy_win": self.spy_win,
                "reason": "Spy made a location guess"
            }
    
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
                # Broadcast the incorrect guess
                broadcast_message = f"{actor} (the spy) guessed the location incorrectly: {location_guess}. The location was {self.location}. The spy loses!"
                await self._broadcast(broadcast_message)