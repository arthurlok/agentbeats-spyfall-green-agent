import random
from pydantic import HttpUrl

class SpyfallEnv:

    def __init__(self, participants: dict[str, HttpUrl], location: str):
        self.participants = participants # name -> agent URL
        self.location = location
        self.round = 0
        self.max_rounds = 0
        self.assigned_roles = {} # similar structure to participants, but instead of storing URLs, store roles
        self.messenger = None # Assume messenger is initialized elsewhere
    
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
    
    async def action_turn(self, name: str, prompt: str) -> str:
        # Logic for a participant to ask a question or guess location (spy only)
        response = await self.messenger.talk_to_agent(
            prompt, str(self.participants[name]), new_conversation=False
        )
        return response

    async def answer_turn(self, name: str, prompt: str) -> str:
        # Logic for a participant to answer a question
        response = await self.messenger.talk_to_agent(
            prompt, str(self.participants[name]), new_conversation=False
        )
        return response

    
    async def play_round(self, assigned_roles: dict[str, str], location: str) -> dict[str, str]:
        if self.round == 0:
            # Initial round logic
            for name in assigned_roles:
                if assigned_roles[name] == "spy":
                    prompt = f"""You are playing a game of Spyfall. Your role is spy. The location is unknown to you but known to the remaining players (non-spies).
                    Each round, all players will take turns asking and answering questions about the location. 
                    Non-spies will attempt to identify the spy while not revealing too much about the location to the spy.
                    As the spy, you must try to identify the location through the conversation and also ask/answer questions without revealing your identity.

                    These are the names of the other players: {', '.join([n for n in assigned_roles if n != name])}.

                    If you feel that you have enough information to guess the location, you may do so on your turn.
                    """
                    response = await self.messenger.talk_to_agent(prompt, str(self.participants[name]), new_conversation=True)
                else:
                    prompt = f"""You are playing a game of Spyfall. Your role is non-spy. The location is {location}.
                    Each round, all non-spy players will take turns asking and answering questions about the location to attempt to identify the spy, who is the only player that does not know the location.
                    As a non-spy, you must try to identify who the spy is through the conversation while answering questions without revealing the location.

                    These are the names of the other players: {', '.join([n for n in assigned_roles if n != name])}.
                    
                    Ask questions strategically to catch the spy, and answer questions honestly to help other non-spies identify the spy.
                    """
                    response = await self.messenger.talk_to_agent(prompt, str(self.participants[name]), new_conversation=True)

            self.round += 1
        # Further rounds logic
        else:
            for name in assigned_roles:
                if assigned_roles[name] == "spy":
                    prompt = f"""It is now your turn to ask a question to a person of your choosing or guess the location.

                    Please respond with a valid JSON object in the following format:
                    {{
                        "action": "<ask_question/guess_location>",
                        "target": "<name_of_participant>" (if action is ask_question),
                        "question": "<your_question_here>" (if action is ask_question),
                        "location_guess": "<your_location_guess_here>" (if action is guess_location)
                    }}
                    """
                    response = await self.messenger.talk_to_agent(prompt, str(self.participants[name]), new_conversation=False)
                else:
                    prompt = f"""It is now your turn to ask a question to a person of your choosing that will help you identify the spy. 

                    Please respond with a valid JSON object in the following format:
                    {{
                        "action": "ask_question",
                        "target": "<name_of_participant>",
                        "question": "<your_question_here>"
                    }}
                    """
                    response = await self.messenger.talk_to_agent(prompt, str(self.participants[name]), new_conversation=False)

            self.round += 1

        return {}