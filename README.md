# Kahoot AI Agent

An automated agent that plays Kahoot games by controlling a web browser, extracting questions and answers, and using OpenAI GPT-4o (or GPT-4o-mini) to select the correct answer.

## Features

- Joins live Kahoot games with a configurable nickname and game PIN
- Waits for each question to appear and extracts question/choice text
- Uses the OpenAI GPT API to answer questions intelligently
- Built with modular, maintainable code using LangGraph, Playwright, and TypedDict state

## Directory Structure

```
kahoot_agent/
├── kahoot_agent/
│   ├── init.py
│   ├── agent.py
│   ├── selectors.py
│   └── state.py
├── .env
├── requirements.txt
├── main.py
└── README.md
```

## Quickstart

### 1. Clone and Install

```
  git clone <>
  cd kahoot_agent
  pip install -r requirements.txt
  playwright install chromium
```

### 2. Set up Environment Variables
Edit `.env` and fill in your credentials:

```
  OPENAI_API_KEY=sk-xxx_your_openai_key
  NICKNAME=AI_Player
  GPT_MODEL=your_model_specification
  KAHOOT_URL=your_kahoot_link
```

### 3. Run the Agent

```
  python main.py
```

The agent will open a browser, join the game, and play automatically!

## Configuration

- **Nickname:** Set via `.env` (`NICKNAME`)
- **OpenAI API:** You must provide your API key in `.env`
- **Selectors:** DOM selectors are kept in `kahoot_agent/selectors.py` for easy updates if Kahoot changes their UI.

## Project Structure

- **kahoot_agent/agent.py**: Main LangGraph workflow and GPT answering logic
- **kahoot_agent/state.py**: TypedDict state definitions
- **kahoot_agent/selectors.py**: HTML selectors for robust scraping
- **main.py**: Script entrypoint
- **.env**: Set your API key, url, and nickname here

## Requirements

- Python 3.9+
- Chrome/Chromium (installed via `playwright install chromium`)
- An OpenAI API key

## Disclaimer

- This is for joining internal TechContest purposes only. Do not use to disrupt real Kahoot sessions or against terms of service.
- Kahoot’s interface may change; update selectors in `kahoot_agent/selectors.py` as needed.
---

Made with ❤️ using [LangGraph](https://github.com/langchain-ai/langgraph), [Playwright](https://playwright.dev/python/), and [OpenAI](https://platform.openai.com/docs/api-reference).
