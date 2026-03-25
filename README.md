# LangGraph router demo

A small demo that routes user prompts with **LangGraph** and **LangChain**: an LLM classifies each message, then one of three agents handles it. A **FastAPI** backend serves a split-pane **HTML** UI (input left, output right).

**Repository:** [github.com/Ritssin/LangGraph](https://github.com/Ritssin/LangGraph)

## How routing works

1. **Router** — `ChatOpenAI` with structured output (`calculation` | `news` | `other`).
2. **Agent 1 (calculation)** — Answers math / numeric questions.
3. **Agent 2 (news)** — Runs a **DuckDuckGo** search on the prompt, then summarizes results with the LLM.
4. **Agent 3 (other)** — Responds with a short **joke** (higher temperature).

Graph flow: `START → router → agent1 | agent2 | agent3 → END`.

## Requirements

- Python 3.10+
- An [OpenAI API](https://platform.openai.com/) key

## Setup

```bash
cd LangGraph
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env and set OPENAI_API_KEY
```

**macOS / Linux:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `OPENAI_MODEL` | No | Defaults to `gpt-4o-mini` |

## Run the app

```bash
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Use the textarea and **Run router**; the API is `POST /api/chat` with JSON body `{"prompt": "your text"}`.

## Project layout

| Path | Role |
|------|------|
| `app/graph.py` | LangGraph definition and agents |
| `app/main.py` | FastAPI app and `/api/chat` |
| `static/index.html` | Web UI |

## Notes

- DuckDuckGo search may occasionally rate-limit or return no results; the news agent still tries to summarize what it gets.
- Do not commit secrets; `.env` is listed in `.gitignore`.

## License

Use and modify as you like for learning and experiments.
