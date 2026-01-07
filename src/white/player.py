import os

from openai import AsyncOpenAI

DEFAULT_MODEL = os.getenv("AGENT_MODEL", "google/gemini-2.0-flash-001")
DEFAULT_SYSTEM_PROMPT = (
    "You are playing a game of Spyfall. Follow the instructions provided by the user exactly. "
    "Keep your statements short and speak in the first person. "
    "When asked to pick a player, respond with only the player's name. "
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


class Player:
    def __init__(self, system_prompt=DEFAULT_SYSTEM_PROMPT, model=DEFAULT_MODEL):
        self.messages = [{"role": "system", "content": system_prompt}]
        self.model = model

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    async def respond(self) -> str:
        response = await client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0,
        )
        statement = (response.choices[0].message.content or "").strip()
        self.add("assistant", statement)
        return statement

    async def handle(self, message: str, skip_response: bool = False) -> str:
        self.add("user", message)
        if skip_response:
            return ""
        return await self.respond()
