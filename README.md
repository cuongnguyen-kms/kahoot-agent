# Kahoot AI Agent

This project is an AI-powered agent for automatically playing Kahoot quizzes. It uses Playwright for browser automation, OpenAI (via LangChain and LangGraph) for question answering, and a set of custom tools for knowledge search, file reading, and image OCR.

## Features

- **Automatic Kahoot gameplay**: Joins a Kahoot game, answers questions, and selects answers in real time.
- **OpenAI LLM integration**: Uses GPT models to answer questions, with a customizable system prompt for smart, context-aware responses.
- **Knowledge base search**: Uses semantic and keyword search over a local knowledge base (`kahoot_knowledge.json`).
- **External file reading**: Can extract and use content from Google Drive and other file links in questions.
- **Image OCR**: Extracts text from images in questions or choices using Tesseract OCR.

## Requirements

- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for image OCR)
- The following Python packages (see `requirements.txt`):
  - playwright
  - openai
  - langgraph
  - python-dotenv
  - sentence-transformers
  - pytesseract
  - Pillow

## Setup

1. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Install Playwright browsers:**

   ```bash
   playwright install
   ```

3. **Install Tesseract OCR:**

   - Windows: Download from [here](https://github.com/tesseract-ocr/tesseract/wiki)
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

4. **Set up environment variables:**

   Create a `.env` file in the project root with:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   KAHOOT_URL=https://kahoot.it/v2/?quizId=...
   NICKNAME=AI_Player
   GPT_MODEL=gpt-4o-mini
   ```

5. **(Optional) Add your knowledge base:**

   - Place a `kahoot_knowledge.json` file in `kahoot_agent/` for custom knowledge search.

## Usage

Run the main script:

```bash
python main.py
```

The agent will join the Kahoot game, answer questions using the LLM and tools, and select answers automatically.

## Tools

- **read_external_file**: Reads and extracts text from external file links (Google Drive, Dropbox, etc.).
- **search_knowledge**: Searches the local knowledge base for relevant information.
- **ocr_image_tool**: Extracts text from images using OCR.

## Customization

- **System Prompt**: Edit `SYSTEM_PROMPT` in `kahoot_agent/constants.py` to change agent behavior.
- **Add More Tools**: Implement new tools in `kahoot_agent/` and add them to the `tools` list in `main.py`.

## Troubleshooting

- If Google Drive file reading fails, ensure the link is public and in a supported format.
- For OCR/image questions, make sure Tesseract is installed and in your system PATH.
- For best LLM results, use the latest GPT models and tune the system prompt as needed.

## License

MIT License
