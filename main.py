import os
import asyncio
from dotenv import load_dotenv

from kahoot_agent.agent import build_agent
from kahoot_agent.state import KahootAgentState

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
NICKNAME = os.environ.get("NICKNAME", "AI_Player")
KAHOOT_URL = os.environ.get("KAHOOT_URL")
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-4o-mini")

async def main():
    agent = build_agent(OPENAI_API_KEY, KAHOOT_URL, GPT_MODEL, NICKNAME)
    initial_state: KahootAgentState = {"step": None}
    await agent.ainvoke(initial_state)
    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(main())