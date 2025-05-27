import os
import asyncio
from dotenv import load_dotenv

from kahoot_agent.agent import kahoot_game_loop_with_react_agent
from kahoot_agent.file_reader_tool import read_external_file
from kahoot_agent.search_tool import search_knowledge
from kahoot_agent.image_tools import ocr_image_tool

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
NICKNAME = os.environ.get("NICKNAME", "AI_Player")
KAHOOT_URL = os.environ.get("KAHOOT_URL")
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-4o-mini")

tools = [read_external_file, search_knowledge, ocr_image_tool]  # Add more tools as needed

async def main():
    await kahoot_game_loop_with_react_agent(
        OPENAI_API_KEY,
        KAHOOT_URL,
        GPT_MODEL,
        NICKNAME,
        tools
    )
    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(main())