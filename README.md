# LectureNotes AI

Convert a lecture PDF (or audio file) into structured notes, flashcards, and a quiz — with automatic factuality verification.

## What it does

| Step | Technology | Type |
|---|---|---|
| PDF text extraction | PyMuPDF | NLP / rule-based |
| Audio transcription | faster-whisper (small) | ML model |
| Topic segmentation | sentence-transformers + numpy cosine | NLP model |
| Keyphrase extraction | KeyBERT | NLP model |
| Named entity recognition | spaCy en_core_web_sm | NLP model |
| Section summaries, flashcards, MCQ | LLM (Azure / Claude / Ollama) | **LLM** |
| Factuality verification | cross-encoder/nli-deberta-v3-base NLI | ML model |
| Multi-language generation | English 🇺🇸 & 日本語 (Japanese) 🇯🇵 | **LLM / i18n** |
| Adaptive difficulty quizzes | Easy, Medium, Hard with live streaks | **Engine** |
| Curated video recommendations | Top educational channels (3Blue1Brown, StatQuest, MIT, Yobinori, etc.) | **Resource Engine** |
| Badges & Progress tracking | XP, Leveling, Concept Mastery, 12 Achievements | **Gamification** |
| Interactive Study Arena | Speed Run, Survival Sudden Death, Hard Gauntlet | **Challenges** |

> **Only step 6 & multi-language generation use an LLM.**  
> Everything else is deterministic NLP, offline rule engines, and local gamification tracking — no API calls required.

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo>
cd lecturenotes-ai
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

**CPU (default):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**GPU (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 4. Configure API keys

```bash
cp .env.example .env
# Edit .env with your preferred provider's key
```

### 5. Set LLM provider in config.yaml

```yaml
llm:
  provider: ollama   # azure | claude | ollama
```

For **Ollama** (free, local): install Ollama and run `ollama pull llama3.1:8b` first.

---

## Run the app

```bash
streamlit run lecturenotes/app/streamlit_app.py
```

Open http://localhost:8501, upload a PDF, and hit **Generate**.

---

## Run evaluation metrics

```bash
python -m lecturenotes.evaluation.run
```

Prints:
- **Factuality rate** — entailed sentences / total (NLI)
- **Concept coverage** — key concepts found in output
- **Pk / WindowDiff** — segmentation quality vs. hand-labeled boundaries
- **Distractor quality** — exports `evaluation_results.csv` for manual rating

---

## Run tests

```bash
pytest tests/ -v
```

---

## Project structure

```
lecturenotes-ai/
├── config.yaml          # All thresholds and model names
├── .env                 # API keys (not committed)
├── requirements.txt
├── lecturenotes/
│   ├── ingest/          # PDF + audio extraction
│   ├── segmentation/    # TextTiling-style topic split
│   ├── concepts/        # KeyBERT + spaCy NER
│   ├── generation/      # LLM wrapper + prompts + Pydantic schemas
│   ├── verification/    # NLI factuality check
│   ├── evaluation/      # Metric reporting CLI
│   └── app/             # Streamlit UI
└── tests/
```

---

## Configuration reference (`config.yaml`)

| Key | Default | Description |
|---|---|---|
| `embedding.model` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `segmentation.threshold` | `0.25` | Cosine-drop threshold for topic boundaries |
| `nli.entailment_threshold` | `0.70` | Below this → flagged in UI |
| `llm.provider` | `ollama` | `azure`, `claude`, or `ollama` |
| `generation.flashcard_count` | `15` | Target flashcard count |
| `generation.mcq_count` | `5` | MCQ questions per lecture |

## Notes

- Python 3.12 is used (3.11-compatible syntax throughout).
- No LangChain, no vector database, no Docker, no auth.
- All expensive operations (embeddings, transcription) are cached to `.cache/diskcache`.
